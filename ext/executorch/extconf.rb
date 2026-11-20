# frozen_string_literal: true

require "fileutils"
require "mkmf"
require "rbconfig"
require "shellwords"
require_relative "../../lib/executorch/sdk"

target_os = RbConfig::CONFIG["target_os"].to_s
is_darwin = target_os.match?(/darwin/i)
is_linux  = target_os.match?(/linux/i)

if target_os.match?(/mswin|mingw|cygwin/i)
  abort "executorch-ruby does not support Windows native builds yet."
end

# ----------------------------------------------------------------------------
# 1. Locate ExecuTorch C++ SDK
# ----------------------------------------------------------------------------

def resolve_dir(raw)
  return nil if raw.nil? || raw.empty?

  expanded = File.expand_path(raw)
  return expanded if Executorch::SDK.valid_sdk?(expanded)

  # Bundler runs extconf.rb inside ext/executorch/; also resolve relative paths
  # against the original invocation PWD if set.
  if !Pathname.new(raw).absolute? && ENV["PWD"]
    from_pwd = File.expand_path(raw, ENV["PWD"])
    return from_pwd if Executorch::SDK.valid_sdk?(from_pwd)
  end

  nil
end

require "pathname"

sdk_dir = nil

if (arg = arg_config("--with-executorch-dir"))
  sdk_dir = resolve_dir(arg)
  unless sdk_dir
    abort <<~ERR
      Invalid --with-executorch-dir: #{arg}
      Expected include/executorch/extension/module/module.h and lib/ inside that directory.
      Tip: use an absolute path (Bundler runs extconf.rb inside ext/executorch/).
    ERR
  end
  puts "Using --with-executorch-dir: #{sdk_dir}"
end

if sdk_dir.nil? && ENV["EXECUTORCH_DIR"] && !ENV["EXECUTORCH_DIR"].empty?
  sdk_dir = resolve_dir(ENV["EXECUTORCH_DIR"])
  unless sdk_dir
    abort "Invalid EXECUTORCH_DIR: #{ENV['EXECUTORCH_DIR']}"
  end
  puts "Using EXECUTORCH_DIR: #{sdk_dir}"
end

if sdk_dir.nil?
  # Check cached official SDK first, then standard system prefixes.
  candidates = [
    Executorch::SDK.default_cache_dir,
    "/opt/homebrew",
    "/usr/local",
  ]
  sdk_dir = candidates.find { |c| Executorch::SDK.valid_sdk?(c) }
end

if sdk_dir.nil? && ENV["EXECUTORCH_NO_DOWNLOAD"] != "1"
  sdk_dir = Executorch::SDK.ensure_sdk
end

unless Executorch::SDK.valid_sdk?(sdk_dir)
  abort <<~ERR
    ExecuTorch C++ SDK not found!
    Provide --with-executorch-dir=/abs/path/to/executorch or EXECUTORCH_DIR=/abs/path,
    or unset EXECUTORCH_NO_DOWNLOAD to fetch the verified ExecuTorch #{Executorch::SDK::VERSION} SDK.
  ERR
end

include_dir = File.join(sdk_dir, "include")
lib_dir     = File.join(sdk_dir, "lib")
c10_inc     = File.join(include_dir, "executorch", "runtime", "core", "portable_type", "c10")

# Read SDK version from executorch.pc if present, else fall back to .executorch-version.
sdk_version = Executorch::SDK::VERSION
pc_file = File.join(lib_dir, "pkgconfig", "executorch.pc")
pc_cflags = []
if File.exist?(pc_file)
  File.readlines(pc_file).each do |line|
    if line =~ /^Version:\s*(\S+)/
      sdk_version = Regexp.last_match(1)
    elsif line =~ /^Cflags:\s*(.+)/
      Regexp.last_match(1).split.each do |tok|
        pc_cflags << tok if tok.start_with?("-D")
      end
    end
  end
end

puts "ExecuTorch SDK: #{sdk_dir} (version #{sdk_version})"

# ----------------------------------------------------------------------------
# 2. Compiler & Symbol Visibility Flags
# ----------------------------------------------------------------------------

$CXXFLAGS = "#{CONFIG['CXXFLAGS']} -std=c++17 -fvisibility=hidden -fvisibility-inlines-hidden"
$CXXFLAGS += " -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations"
$CXXFLAGS += " -DEXECUTORCH_RUBY_SDK_VERSION=\\\"#{sdk_version}\\\""

if pc_cflags.any?
  pc_cflags.each { |flag| $CXXFLAGS += " #{flag}" }
else
  $CXXFLAGS += " -DC10_USING_CUSTOM_GENERATED_MACROS"
end

if ENV["EXECUTORCH_SANITIZE"] == "1"
  puts "Enabling AddressSanitizer + UndefinedBehaviorSanitizer"
  $CXXFLAGS += " -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g"
  $LDFLAGS  += " -fsanitize=address,undefined"
end

$INCFLAGS = "-I#{Shellwords.escape(include_dir)} #{$INCFLAGS}"
if File.directory?(c10_inc)
  $INCFLAGS = "-I#{Shellwords.escape(c10_inc)} #{$INCFLAGS}"
end
$INCFLAGS = "-I#{Shellwords.escape(File.join(__dir__, 'core'))} #{$INCFLAGS}"

# ----------------------------------------------------------------------------
# 3. Linking: Shared SDK (preferred) vs Static Source Build
# ----------------------------------------------------------------------------

dlext = is_darwin ? "dylib" : "so"
shared_runtime = File.join(lib_dir, "libexecutorch.#{dlext}")

bundle_vendor = ENV["EXECUTORCH_BUNDLE_VENDOR"] == "1"
vendor_dir = File.expand_path("../../lib/executorch/vendor", __dir__)

$LDFLAGS += " -L#{Shellwords.escape(lib_dir)}"

if File.exist?(shared_runtime)
  puts "Linking mode: shared ExecuTorch SDK (.#{dlext})"

  if bundle_vendor
    FileUtils.mkdir_p(vendor_dir)
    key = Executorch::SDK.platform_key
    files = Executorch::SDK::VENDOR_LIBS[key] || Dir.glob(File.join(lib_dir, "libexecutorch*.#{dlext}"))
                                                    .map { |p| File.basename(p) }
    files.each do |fname|
      src = File.join(lib_dir, fname)
      next unless File.exist?(src)

      FileUtils.cp(src, File.join(vendor_dir, fname))
      puts "  Vendored: #{fname}"
    end
  end

  if is_darwin
    if bundle_vendor
      $LDFLAGS += " -Wl,-rpath,@loader_path/vendor -Wl,-rpath,@loader_path/../vendor"
    else
      $LDFLAGS += " -Wl,-rpath,#{Shellwords.escape(lib_dir)}"
    end
  elsif is_linux
    if bundle_vendor
      $LDFLAGS += " -Wl,--enable-new-dtags -Wl,-rpath,'$$ORIGIN/vendor' -Wl,-rpath,'$$ORIGIN/../vendor'"
    else
      $LDFLAGS += " -Wl,--enable-new-dtags -Wl,-rpath,#{Shellwords.escape(lib_dir)}"
    end
  end

  # Core runtime + threadpool
  $LOCAL_LIBS += " -lexecutorch"
  if File.exist?(File.join(lib_dir, "libexecutorch_threadpool.#{dlext}"))
    $LOCAL_LIBS += " -lexecutorch_threadpool"
    $CXXFLAGS += " -DET_USE_THREADPOOL" unless $CXXFLAGS.include?("-DET_USE_THREADPOOL")
  end

  # Self-registering shared kernel & backend plugins:
  # Must NOT be dropped by linker --as-needed / -dead_strip_dylibs!
  plugin_libs = %w[
    executorch_kernels_optimized
    executorch_kernels_quantized
    executorch_kernels_torchao
    executorch_backend_xnnpack
    executorch_backend_coreml
    executorch_backend_mlx
  ].select { |name| File.exist?(File.join(lib_dir, "lib#{name}.#{dlext}")) }

  unless plugin_libs.empty?
    puts "  Plugins: #{plugin_libs.join(', ')}"
    if is_linux
      $LOCAL_LIBS += " -Wl,--no-as-needed"
      plugin_libs.each { |lib| $LOCAL_LIBS += " -l#{lib}" }
      $LOCAL_LIBS += " -Wl,--as-needed"
    elsif is_darwin
      plugin_libs.each { |lib| $LOCAL_LIBS += " -Wl,-needed-l#{lib}" }
    end
  end
else
  # Fallback: static ExecuTorch source build (.a archives)
  puts "Linking mode: static ExecuTorch archives (.a)"

  def force_load_flag(path, is_darwin)
    escaped = Shellwords.escape(path)
    if is_darwin
      " -Wl,-force_load,#{escaped}"
    else
      " -Wl,--whole-archive #{escaped} -Wl,--no-whole-archive"
    end
  end

  mandatory = %w[extension_module_static extension_data_loader extension_tensor executorch executorch_core]
  mandatory.each do |lib|
    path = File.join(lib_dir, "lib#{lib}.a")
    alt  = File.join(lib_dir, "lib#{lib.sub(/_static$/, '')}.a")
    unless File.exist?(path) || File.exist?(alt)
      abort "Required static library missing: #{lib} (checked #{path})"
    end
  end

  whole_libs = []
  # Prefer optimized_native_cpu_ops_lib over portable_ops_lib (mutually exclusive)
  %w[optimized_native_cpu_ops_lib portable_ops_lib].each do |ops|
    p = File.join(lib_dir, "lib#{ops}.a")
    if File.exist?(p)
      whole_libs << p
      break
    end
  end
  # Additional non-conflicting self-registering archives
  %w[quantized_ops_lib xnnpack_backend coreml_backend mps_backend].each do |extra|
    p = File.join(lib_dir, "lib#{extra}.a")
    whole_libs << p if File.exist?(p)
  end

  whole_libs.each do |p|
    puts "  Whole-archive: #{File.basename(p)}"
    $LOCAL_LIBS += force_load_flag(p, is_darwin)
  end

  # Static dependency archives (ordered & grouped on Linux so GNU ld resolves cycles cleanly)
  static_deps = %w[
    extension_module_static extension_module
    extension_data_loader extension_tensor extension_named_data_map
    extension_flat_tensor extension_threadpool
    optimized_kernels portable_kernels quantized_kernels cpublas eigen_blas
    XNNPACK xnnpack-microkernels-prod microkernels-prod kleidiai
    pthreadpool cpuinfo clog
    executorch executorch_core
  ].uniq.map { |l| File.join(lib_dir, "lib#{l}.a") }.select { |p| File.exist?(p) }

  $LOCAL_LIBS += " -Wl,--start-group" if is_linux
  static_deps.each { |p| $LOCAL_LIBS += " #{Shellwords.escape(p)}" }
  $LOCAL_LIBS += " -Wl,--end-group" if is_linux

  if is_darwin
    $LOCAL_LIBS += " -framework Accelerate"
    if File.exist?(File.join(lib_dir, "libcoreml_backend.a"))
      $LOCAL_LIBS += " -framework CoreML -framework Foundation -lsqlite3"
    end
  end
end

# Export ONLY Init_executorch so zero C++ / ExecuTorch symbols leak globally!
if is_linux
  map_file = File.expand_path("executorch.map", __dir__)
  $DLDFLAGS += " -Wl,--version-script=#{Shellwords.escape(map_file)} -Wl,--exclude-libs,ALL"
  $LOCAL_LIBS += " -lpthread -lm -ldl"
elsif is_darwin
  $DLDFLAGS += " -Wl,-exported_symbol,_Init_executorch"
end

# ----------------------------------------------------------------------------
# 4. Sources & Makefile
# ----------------------------------------------------------------------------

$VPATH << "$(srcdir)/core"
$srcs = %w[
  core.cpp
  rb_errors.cpp
  rb_tensor.cpp
  rb_module.cpp
  rb_executorch.cpp
]

create_makefile("executorch/executorch")
