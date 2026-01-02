# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)

require "executorch"
require "minitest/autorun"

module TestHelper
  # Path to test models directory
  def models_dir
    File.expand_path("support/models", __dir__)
  end

  # Path to a specific test model
  def model_path(name)
    File.join(models_dir, name)
  end

  # Check if test models are available
  def models_available?
    File.directory?(models_dir) && !Dir.glob(File.join(models_dir, "*.pte")).empty?
  end

  # Skip test if models are not available
  def skip_without_models
    skip "Test models not available. See test/support/README.md" unless models_available?
  end
end
