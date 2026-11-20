# frozen_string_literal: true

require "bundler/gem_tasks"
require "fileutils"
require "rake/extensiontask"
require "rake/testtask"
require "rbconfig"
require_relative "lib/executorch/sdk"

gemspec = Gem::Specification.load(File.expand_path("executorch.gemspec", __dir__))

Rake::ExtensionTask.new("executorch", gemspec) do |ext|
  ext.lib_dir = "lib/executorch"
  ext.source_pattern = "*.{cpp,hpp,h}"
end

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/*_test.rb"]
  t.warning = true
end

task default: %i[compile test]

namespace :sdk do
  desc "Download and verify the pinned ExecuTorch C++ SDK into local cache"
  task :fetch do
    dir = Executorch::SDK.ensure_sdk
    puts "SDK ready at: #{dir}"
  end
end

namespace :fixtures do
  desc "Export deterministic .pte/.ptd test fixtures using uv"
  task :export do
    out_dir = ENV.fetch("EXECUTORCH_FIXTURES_DIR", File.expand_path("test/fixtures/models", __dir__))
    py = ENV["EXECUTORCH_PYTHON"] || File.expand_path("~/.cache/executorch-ruby/venv/bin/python")
    if File.executable?(py)
      sh py, "test/fixtures/export_fixtures.py", "--out", out_dir
    else
      sh "uv", "run", "--project", "test/fixtures", "python", "test/fixtures/export_fixtures.py", "--out", out_dir
    end
  end
end

desc "Clean build outputs (preserves SDK cache and exported fixtures)"
task :clean do
  dlext = RbConfig::CONFIG["DLEXT"]
  FileUtils.rm_rf("tmp/x86_64-linux")
  FileUtils.rm_rf("tmp/aarch64-linux")
  FileUtils.rm_rf("tmp/arm64-darwin")
  FileUtils.rm_rf("tmp/x86_64-darwin")
  FileUtils.rm_rf("pkg")
  FileUtils.rm_rf("lib/executorch/vendor")
  Dir.glob("lib/executorch/**/*.{#{dlext},so,bundle}").each { |f| FileUtils.rm_f(f) }
end

desc "Build a precompiled platform fat gem for the current host platform"
task :package_platform do
  platform = Gem::Platform.local
  ENV["EXECUTORCH_BUNDLE_VENDOR"] = "1"
  Rake::Task[:clean].invoke
  Rake::Task[:compile].invoke

  ruby_abi = RUBY_VERSION[/\A\d+\.\d+/]
  dlext = RbConfig::CONFIG["DLEXT"]
  src_ext = "lib/executorch/executorch.#{dlext}"
  abi_dir = "lib/executorch/#{ruby_abi}"
  FileUtils.mkdir_p(abi_dir)
  FileUtils.mv(src_ext, File.join(abi_dir, "executorch.#{dlext}"))

  plat_spec = gemspec.dup
  plat_spec.platform = platform
  plat_spec.extensions = []
  plat_spec.files = (
    gemspec.files.reject { |f| f.start_with?("ext/") } +
    Dir["lib/executorch/#{ruby_abi}/*.#{dlext}"] +
    Dir["lib/executorch/vendor/*"]
  ).uniq.sort

  FileUtils.mkdir_p("pkg")
  gem_file = Gem::Package.build(plat_spec)
  FileUtils.mv(gem_file, File.join("pkg", File.basename(gem_file)))
  puts "Built platform gem: pkg/#{File.basename(gem_file)}"
end
