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
  desc 'Build ExecuTorch from source'
  task :build_deps do
    executorch_dir = ENV.fetch('EXECUTORCH_SRC', File.expand_path('$HOME/.local/executorch'))

    unless File.directory?(executorch_dir)
      abort "ExecuTorch source not found at #{executorch_dir}. Set EXECUTORCH_SRC environment variable."
    end

    puts "Building ExecuTorch at #{executorch_dir}..."

    Dir.chdir(executorch_dir) do
      system('cmake', '-B', 'cmake-out',
             "-DCMAKE_INSTALL_PREFIX=#{executorch_dir}/install",
             '-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON',
             '-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON',
             '-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON',
             '-DCMAKE_BUILD_TYPE=Release') || abort('CMake configure failed')

      system('cmake', '--build', 'cmake-out', "-j#{Etc.nprocessors}") || abort('CMake build failed')
      system('cmake', '--install', 'cmake-out') || abort('CMake install failed')
    end

    puts "ExecuTorch built successfully at #{executorch_dir}/install"
  end
end

desc 'Clean build artifacts'
task :clean do
  FileUtils.rm_rf('lib/executorch/executorch.bundle')
  FileUtils.rm_rf('lib/executorch/executorch.so')
  FileUtils.rm_rf('tmp')
end
