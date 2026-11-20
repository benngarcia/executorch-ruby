# frozen_string_literal: true

require_relative "lib/executorch/version"

Gem::Specification.new do |spec|
  spec.name                  = "executorch"
  spec.version               = Executorch::VERSION
  spec.authors               = ["Benjamin Garcia"]
  spec.email                 = ["hey@bengarcia.dev"]

  spec.summary               = "Native Ruby bindings for Meta's ExecuTorch runtime"
  spec.description           = "Run PyTorch models exported with ExecuTorch (.pte / .ptd) natively in Ruby with XNNPACK/CoreML acceleration, GVL-free execution, and zero-copy IO::Buffer tensors."
  spec.homepage              = "https://github.com/benngarcia/executorch-ruby"
  spec.license               = "Apache-2.0"
  spec.required_ruby_version = ">= 3.2.0"

  spec.files = Dir.chdir(__dir__) do
    if File.directory?(".git")
      `git ls-files -z`.split("\x0").select do |f|
        f.start_with?("lib/", "ext/") ||
          %w[.executorch-version LICENSE.txt README.md CHANGELOG.md].include?(f)
      end
    else
      Dir[
        ".executorch-version",
        "LICENSE.txt",
        "README.md",
        "CHANGELOG.md",
        "lib/**/*.rb",
        "ext/**/*.{rb,cpp,hpp,h,map}",
      ]
    end
  end

  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/executorch/extconf.rb"]

  spec.metadata = {
    "source_code_uri"       => spec.homepage,
    "changelog_uri"         => "#{spec.homepage}/blob/master/CHANGELOG.md",
    "bug_tracker_uri"       => "#{spec.homepage}/issues",
    "rubygems_mfa_required" => "true",
  }
end
