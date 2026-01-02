require 'bundler/gem_tasks'
require 'rake/testtask'
require 'rake/extensiontask'

Rake::TestTask.new(:test) do |t|
  t.libs << 'test'
  t.libs << 'lib'
  t.test_files = FileList['test/**/*_test.rb']
end

Rake::ExtensionTask.new('executorch') do |ext|
  ext.lib_dir = 'lib/executorch'
end

task default: %i[compile test]

namespace :executorch do
  desc 'Build ExecuTorch from source and install to vendor/executorch'
  task :build_deps do
    source_dir = ENV.fetch('EXECUTORCH_SRC', File.expand_path('../executorch', __dir__))
    install_dir = File.expand_path('vendor/executorch', __dir__)

    unless File.directory?(source_dir)
      abort "ExecuTorch source not found at #{source_dir}. Clone it or set EXECUTORCH_SRC."
    end

    puts "Building ExecuTorch from #{source_dir}..."
    puts "Installing to #{install_dir}..."

    Dir.chdir(source_dir) do
      system('cmake', '-B', 'cmake-out',
             "-DCMAKE_INSTALL_PREFIX=#{install_dir}",
             '-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON',
             '-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON',
             '-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON',
             '-DCMAKE_BUILD_TYPE=Release') || abort('CMake configure failed')

      system('cmake', '--build', 'cmake-out', "-j#{Etc.nprocessors}") || abort('CMake build failed')
      system('cmake', '--install', 'cmake-out') || abort('CMake install failed')
    end

    puts "ExecuTorch installed to #{install_dir}"
    puts "Run: bundle config set --local build.executorch --with-executorch-dir=#{install_dir}"
  end
end

desc 'Clean build artifacts'
task :clean do
  FileUtils.rm_rf('lib/executorch/executorch.bundle')
  FileUtils.rm_rf('lib/executorch/executorch.so')
  FileUtils.rm_rf('tmp')
end
