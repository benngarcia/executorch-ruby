# frozen_string_literal: true

require "digest"
require "fileutils"
require "net/http"
require "rbconfig"
require "tmpdir"
require "uri"
require "zlib"

module Executorch
  # Fetches and extracts the official prebuilt ExecuTorch C++ SDK (headers,
  # shared runtime/backend/kernel libraries, pkg-config, and CMake configs)
  # from official PyPI wheels with strict SHA-256 verification and zero
  # non-stdlib dependencies.
  module SDK
    VERSION = File.read(File.expand_path("../../.executorch-version", __dir__)).strip.freeze

    # Official ExecuTorch 1.6.0 PyPI wheel manifest (only include/, lib/, share/cmake/ are extracted).
    MANIFEST = {
      "x86_64-linux" => {
        url: "https://files.pythonhosted.org/packages/ef/24/f8fde978e1e5fd629d68c74a3f4acee30acb05f4f90c433cc1a8e29d8eed/executorch-1.6.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        sha256: "d05efe9afe00495112ecf304ca548fcd3239d39e96ac25381b528cdbad948b54",
      },
      "aarch64-linux" => {
        url: "https://files.pythonhosted.org/packages/d5/7f/9028996ec1dead9c57ed2c6384037e3567c2f514e4ed0c54480d2d3eeb48/executorch-1.6.0-cp312-cp312-manylinux_2_28_aarch64.whl",
        sha256: "ff595d236f4052a196a03a867ca0f53d2e3b003d00ab871433660360e194e78e",
      },
      "arm64-darwin" => {
        url: "https://files.pythonhosted.org/packages/c6/47/1e2120ca1ee178e81471d57255ea0f87b6e04272e32ba20e5ed60ebdb46d/executorch-1.6.0-cp312-cp312-macosx_14_0_arm64.whl",
        sha256: "02c220037b41a0a6074a94499a747400ef1a5075a3f2b4d83911871c7291a341",
      },
    }.freeze

    # Core runtime + CPU/hardware delegate shared libs bundled in fat gems.
    VENDOR_LIBS = {
      "x86_64-linux" => %w[
        libexecutorch.so
        libexecutorch_threadpool.so
        libexecutorch_kernels_optimized.so
        libexecutorch_kernels_quantized.so
        libexecutorch_backend_xnnpack.so
        libexecutorch_etdump.so
      ],
      "aarch64-linux" => %w[
        libexecutorch.so
        libexecutorch_threadpool.so
        libexecutorch_kernels_optimized.so
        libexecutorch_kernels_quantized.so
        libexecutorch_kernels_torchao.so
        libexecutorch_backend_xnnpack.so
        libexecutorch_etdump.so
      ],
      "arm64-darwin" => %w[
        libexecutorch.dylib
        libexecutorch_threadpool.dylib
        libexecutorch_kernels_optimized.dylib
        libexecutorch_kernels_quantized.dylib
        libexecutorch_kernels_torchao.dylib
        libexecutorch_backend_xnnpack.dylib
        libexecutorch_backend_coreml.dylib
        libexecutorch_backend_mlx.dylib
        libexecutorch_etdump.dylib
        mlx.metallib
      ],
    }.freeze

    module_function

    def platform_key(os: RbConfig::CONFIG["target_os"], cpu: RbConfig::CONFIG["target_cpu"])
      norm_cpu = case cpu
                 when /x86_64|amd64/i then "x86_64"
                 when /aarch64|arm64/i then os =~ /darwin/i ? "arm64" : "aarch64"
                 else cpu
                 end
      norm_os = case os
                when /darwin/i then "darwin"
                when /linux/i then "linux"
                else os
                end
      "#{norm_cpu}-#{norm_os}"
    end

    def default_cache_dir(key = platform_key)
      base = ENV["XDG_CACHE_HOME"]
      base = File.join(Dir.home, ".cache") if base.nil? || base.empty?
      File.join(base, "executorch-ruby", "sdk-#{VERSION}-#{key}")
    end

    def valid_sdk?(dir)
      return false unless dir && File.directory?(dir)

      File.exist?(File.join(dir, "include", "executorch", "extension", "module", "module.h")) &&
        File.directory?(File.join(dir, "lib"))
    end

    def ensure_sdk(dest_dir: nil, key: platform_key, io: $stdout)
      dest = dest_dir || default_cache_dir(key)
      return dest if valid_sdk?(dest)

      entry = MANIFEST[key]
      raise "No prebuilt ExecuTorch #{VERSION} SDK manifest for platform '#{key}'. Set --with-executorch-dir or EXECUTORCH_DIR." unless entry

      io&.puts "Fetching official ExecuTorch #{VERSION} C++ SDK for #{key}..."
      Dir.mktmpdir("executorch-sdk") do |tmp|
        whl_path = File.join(tmp, "executorch.whl")
        download_with_redirects(entry[:url], whl_path)
        actual_sha = Digest::SHA256.file(whl_path).hexdigest
        unless actual_sha == entry[:sha256]
          raise "SHA-256 mismatch for ExecuTorch SDK (#{key}): expected #{entry[:sha256]}, got #{actual_sha}"
        end

        stage = File.join(tmp, "stage")
        extract_sdk_from_wheel(whl_path, stage)
        FileUtils.mkdir_p(File.dirname(dest))
        FileUtils.rm_rf(dest)
        FileUtils.mv(stage, dest)
      end

      raise "Extracted SDK at #{dest} is incomplete" unless valid_sdk?(dest)

      io&.puts "Installed ExecuTorch #{VERSION} C++ SDK to #{dest}"
      dest
    end

    def download_with_redirects(url, dest, limit = 5)
      raise "Too many HTTP redirects downloading #{url}" if limit <= 0

      uri = URI.parse(url)
      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", open_timeout: 30, read_timeout: 120) do |http|
        req = Net::HTTP::Get.new(uri)
        http.request(req) do |res|
          case res
          when Net::HTTPSuccess
            File.open(dest, "wb") { |f| res.read_body { |chunk| f.write(chunk) } }
          when Net::HTTPRedirection
            download_with_redirects(res["location"], dest, limit - 1)
          else
            raise "Failed downloading #{url}: HTTP #{res.code} #{res.message}"
          end
        end
      end
    end

    # Minimal pure-Ruby ZIP reader (central directory) extracting ONLY
    # executorch/{include,lib,share/cmake}/ from the wheel.
    def extract_sdk_from_wheel(whl_path, dest_dir)
      File.open(whl_path, "rb") do |f|
        size = f.size
        search_len = [size, 65_557].min
        f.seek(size - search_len)
        tail = f.read(search_len)
        eocd_idx = tail.rindex("PK\x05\x06")
        raise "Invalid wheel archive (missing EOCD)" unless eocd_idx

        _sig, _disk, _cd_disk, _entries_disk, total_entries, cd_size, cd_offset =
          tail.byteslice(eocd_idx, 22).unpack("a4vvvvVV")

        # Handle ZIP64 if needed
        if total_entries == 0xFFFF || cd_offset == 0xFFFFFFFF
          loc_idx = tail.rindex("PK\x06\x07", eocd_idx)
          raise "ZIP64 locator missing" unless loc_idx

          zip64_offset = tail.byteslice(loc_idx + 8, 8).unpack1("Q<")
          f.seek(zip64_offset)
          z64 = f.read(56)
          total_entries = z64.byteslice(32, 8).unpack1("Q<")
          cd_size = z64.byteslice(40, 8).unpack1("Q<")
          cd_offset = z64.byteslice(48, 8).unpack1("Q<")
        end

        f.seek(cd_offset)
        cd = f.read(cd_size)
        pos = 0

        total_entries.times do
          break if pos + 46 > cd.bytesize

          sig = cd.byteslice(pos, 4)
          raise "Invalid central directory header" unless sig == "PK\x01\x02"

          method, _time, _date, _crc, comp_size, uncomp_size, name_len, extra_len, comment_len,
            _disk_start, _int_attr, _ext_attr, local_offset =
            cd.byteslice(pos + 10, 36).unpack("vvvVVVvvvvvVV")

          name = cd.byteslice(pos + 46, name_len)
          extra = cd.byteslice(pos + 46 + name_len, extra_len)
          pos += 46 + name_len + extra_len + comment_len

          next unless name.start_with?("executorch/include/", "executorch/lib/", "executorch/share/cmake/")
          next if name.end_with?("/")

          # Resolve ZIP64 extra fields if 32-bit fields overflowed
          if comp_size == 0xFFFFFFFF || uncomp_size == 0xFFFFFFFF || local_offset == 0xFFFFFFFF
            ep = 0
            while ep + 4 <= extra.bytesize
              eid, esz = extra.byteslice(ep, 4).unpack("vv")
              if eid == 0x0001
                ef = extra.byteslice(ep + 4, esz)
                off = 0
                if uncomp_size == 0xFFFFFFFF
                  uncomp_size = ef.byteslice(off, 8).unpack1("Q<")
                  off += 8
                end
                if comp_size == 0xFFFFFFFF
                  comp_size = ef.byteslice(off, 8).unpack1("Q<")
                  off += 8
                end
                if local_offset == 0xFFFFFFFF
                  local_offset = ef.byteslice(off, 8).unpack1("Q<")
                end
                break
              end
              ep += 4 + esz
            end
          end

          rel = name.delete_prefix("executorch/")
          # Prevent path traversal
          raise "Unsafe archive path: #{rel}" if rel.include?("..") || rel.start_with?("/")

          out_path = File.join(dest_dir, rel)
          FileUtils.mkdir_p(File.dirname(out_path))

          f.seek(local_offset)
          lh = f.read(30)
          raise "Invalid local file header for #{name}" unless lh&.start_with?("PK\x03\x04")

          l_name_len, l_extra_len = lh.byteslice(26, 4).unpack("vv")
          f.seek(local_offset + 30 + l_name_len + l_extra_len)
          raw = f.read(comp_size)

          data = case method
                 when 0 then raw
                 when 8
                   z = Zlib::Inflate.new(-Zlib::MAX_WBITS)
                   begin
                     z.inflate(raw)
                   ensure
                     z.close
                   end
                 else
                   raise "Unsupported ZIP compression method #{method} for #{name}"
                 end

          File.binwrite(out_path, data)
        end
      end
    end
  end
end
