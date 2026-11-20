# frozen_string_literal: true

require_relative "executorch/version"
require_relative "executorch/method_meta"

# Load precompiled Ruby-version-specific native extension first (fat gems),
# falling back to standard single-version native extension path.
begin
  ruby_abi = RUBY_VERSION[/\A\d+\.\d+/]
  require_relative "executorch/#{ruby_abi}/executorch"
rescue LoadError
  require_relative "executorch/executorch"
end

require_relative "executorch/tensor"
require_relative "executorch/module"
