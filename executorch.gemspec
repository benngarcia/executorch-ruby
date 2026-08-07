require_relative 'lib/executorch/version'

Gem::Specification.new do |spec|
  spec.name          = 'executorch'
  spec.version       = Executorch::VERSION
  spec.authors       = ['Benjamin Garcia']
  spec.email         = ['hey@bengarcia.dev']

  spec.summary       = 'Ruby bindings for ExecuTorch'
  spec.description   = 'Run PyTorch models exported with ExecuTorch in Ruby'
  spec.homepage      = 'https://github.com/benngarcia/executorch-ruby'
  spec.license       = 'Apache-2.0'
  spec.required_ruby_version = '>= 3.0.0'

  spec.files = Dir[
    'LICENSE.txt',
    'README.md',
    'CHANGELOG.md',
    'lib/**/*.rb',
    'ext/**/*.{rb,cpp,h,hpp}'
  ]

  spec.require_paths = ['lib']
  spec.extensions    = ['ext/executorch/extconf.rb']

  spec.add_dependency 'rice', '>= 4.3'

  spec.metadata = {
    'homepage_uri' => spec.homepage,
    'source_code_uri' => spec.homepage,
    'changelog_uri' => "#{spec.homepage}/blob/master/CHANGELOG.md"
  }
end
