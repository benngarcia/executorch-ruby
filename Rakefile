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
    # Delegates to the script so this task, the README, and CI all build
    # ExecuTorch exactly the same way. Set EXECUTORCH_BACKENDS=xnnpack to
    # include the delegate.
    sh File.expand_path('script/build-executorch.sh', __dir__)
  end
end

namespace :bench do
  models_dir = File.expand_path('bench/models', __dir__)

  desc 'Export the .pt benchmark checkpoints and convert them to .pte'
  task :prepare do
    sh 'python3', 'bench/make_pt_models.py', '--all'
    sh "python3 bench/pt_to_pte.py #{models_dir}/*.pt"
  end

  desc 'Run evals + profiling (LABEL=name to tag the results file)'
  task run: :compile do
    sh 'ruby', 'bench/run_bench.rb', '--label', ENV.fetch('LABEL', 'current')
  end

  desc 'Compare two result files: rake bench:compare BASE=baseline CURRENT=mine'
  task :compare do
    base = ENV.fetch('BASE', 'baseline')
    current = ENV.fetch('CURRENT', 'current')
    sh 'ruby', 'bench/compare.rb',
       "bench/results/#{base}.json", "bench/results/#{current}.json"
  end
end

desc 'Clean build artifacts'
task :clean do
  FileUtils.rm_rf('lib/executorch/executorch.bundle')
  FileUtils.rm_rf('lib/executorch/executorch.so')
  FileUtils.rm_rf('tmp')
end
