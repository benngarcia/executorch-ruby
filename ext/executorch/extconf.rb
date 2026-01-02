require 'mkmf-rice'

$CXXFLAGS += ' -std=c++17'
$CXXFLAGS += ' -DC10_USING_CUSTOM_GENERATED_MACROS'
$CXXFLAGS += ' -Wno-deprecated-declarations'

# ==============================================================================
# ExecuTorch Path Detection
# ==============================================================================
#
# 1. --with-executorch-dir flag (bundle config or gem install)
# 2. EXECUTORCH_DIR environment variable
#

include_dirs = []
lib_dirs = []

# Helper to add prefix paths
def add_prefix_paths(prefix, include_dirs, lib_dirs)
  return false unless prefix && File.directory?(prefix)

  inc = File.join(prefix, 'include')
  lib = File.join(prefix, 'lib')

  if File.directory?(inc) && File.directory?(lib)
    include_dirs << inc
    lib_dirs << lib
    true
  else
    false
  end
end

# Priority 1: --with-executorch-dir flag (standard Ruby gem pattern)
# Usage: bundle config set --local build.executorch --with-executorch-dir=/path/to/executorch
#    or: gem install executorch -- --with-executorch-dir=/path/to/executorch
executorch_dir = arg_config('--with-executorch-dir')
if executorch_dir
  if add_prefix_paths(executorch_dir, include_dirs, lib_dirs)
    puts "Using --with-executorch-dir: #{executorch_dir}"
  else
    abort "Error: --with-executorch-dir path is invalid: #{executorch_dir}"
  end
end

# Priority 2: Environment variable (for CI/scripting)
if include_dirs.empty?
  executorch_prefix = ENV['EXECUTORCH_DIR']
  if add_prefix_paths(executorch_prefix, include_dirs, lib_dirs)
    puts "Using EXECUTORCH_DIR: #{executorch_prefix}"
  end
end

include_dirs.compact!
include_dirs.uniq!
lib_dirs.compact!
lib_dirs.uniq!

if include_dirs.empty? || lib_dirs.empty?
  abort <<~ERROR
    ExecuTorch installation not found!

    Configure the path using one of these methods:

    1. Bundle config (recommended - set once per project):
       bundle config set --local build.executorch --with-executorch-dir=/path/to/executorch

    2. Environment variable (useful for CI):
       EXECUTORCH_DIR=/path/to/executorch bundle install

    Need to build ExecuTorch first? See: https://pytorch.org/executorch/
  ERROR
end

# Validate headers exist
include_dir = include_dirs.first
lib_dir = lib_dirs.first

unless File.exist?(File.join(include_dir, 'executorch', 'extension', 'module', 'module.h'))
  abort <<~ERROR
    ExecuTorch module.h header not found at: #{include_dir}
    Make sure EXECUTORCH_BUILD_EXTENSION_MODULE=ON was set during build.
  ERROR
end

puts "Include directory: #{include_dir}"
puts "Library directory: #{lib_dir}"

# Configure include paths
# Also add the portable c10 headers path for standalone builds
portable_c10_dir = File.join(include_dir, 'executorch', 'runtime', 'core', 'portable_type', 'c10')
$INCFLAGS = "-I#{include_dir} -I#{portable_c10_dir} #{$INCFLAGS}"

# Configure library paths
$LDFLAGS += " -L#{lib_dir}"

# Add rpath for runtime library loading
$LDFLAGS += if RUBY_PLATFORM =~ /darwin/
              " -Wl,-rpath,#{lib_dir}"
            else
              " -Wl,-rpath,#{lib_dir}"
            end

# ==============================================================================
# Library Linking Configuration
# ==============================================================================

# Default libraries required for basic operation
DEFAULT_LIBS = %w[
  extension_module_static
  extension_data_loader
  extension_tensor
  extension_named_data_map
  extension_flat_tensor
  extension_threadpool
  executorch
  executorch_core
].freeze

# Determine which libraries to link
libs = if ENV['EXECUTORCH_LIBS']
         # User-specified library list (overrides defaults)
         user_libs = ENV['EXECUTORCH_LIBS'].split(',').map(&:strip).reject(&:empty?)
         puts "Using custom library list: #{user_libs.join(', ')}"
         user_libs
       else
         DEFAULT_LIBS.dup
       end

# Link extension libraries first (order matters for static linking)
libs.each do |lib|
  lib_file = File.join(lib_dir, "lib#{lib}.a")
  if File.exist?(lib_file)
    $LDFLAGS += " -l#{lib}"
    puts "  Linking: #{lib}"
  else
    # Try without _static suffix
    alt_lib = lib.sub(/_static$/, '')
    alt_file = File.join(lib_dir, "lib#{alt_lib}.a")
    if File.exist?(alt_file)
      $LDFLAGS += " -l#{alt_lib}"
      puts "  Linking: #{alt_lib} (alternative)"
    elsif lib.include?('extension_')
      # Extension libraries are optional
      puts "  Skipping optional: #{lib} (not found)"
    else
      # Core libraries should exist
      abort "Required library not found: #{lib} (looked for #{lib_file})"
    end
  end
end

# Check for portable kernels (if available)
# Use -force_load on macOS to include global constructors that register kernels
portable_ops_lib = File.join(lib_dir, 'libportable_ops_lib.a')
if File.exist?(portable_ops_lib)
  $LDFLAGS += if RUBY_PLATFORM =~ /darwin/
                # macOS: -force_load pulls in all symbols including global constructors
                " -Wl,-force_load,#{portable_ops_lib}"
              else
                # Linux: --whole-archive achieves the same effect
                ' -Wl,--whole-archive -lportable_ops_lib -Wl,--no-whole-archive'
              end
  puts '  Linking: portable_ops_lib (force_load)'

  # Portable kernels
  if File.exist?(File.join(lib_dir, 'libportable_kernels.a'))
    $LDFLAGS += ' -lportable_kernels'
    puts '  Linking: portable_kernels'
  end
end

# CPU info and pthreadpool (often required)
%w[cpuinfo pthreadpool].each do |lib|
  if File.exist?(File.join(lib_dir, "lib#{lib}.a"))
    $LDFLAGS += " -l#{lib}"
    puts "  Linking: #{lib}"
  end
end

# Extra libraries from environment (consolidated from version B)
if ENV['EXECUTORCH_EXTRA_LIBS']
  extra_libs = ENV['EXECUTORCH_EXTRA_LIBS'].split(',').map(&:strip).reject(&:empty?)
  extra_libs.each do |lib|
    if File.exist?(File.join(lib_dir, "lib#{lib}.a"))
      $LDFLAGS += " -l#{lib}"
      puts "  Linking extra: #{lib}"
    else
      puts "  Warning: extra library not found: #{lib}"
    end
  end
end

# ==============================================================================
# Platform-Specific Configuration
# ==============================================================================

if (RUBY_PLATFORM =~ /darwin/) && (RUBY_PLATFORM =~ /arm64/)
  # Apple Silicon specific
  puts 'Building for Apple Silicon (arm64)'
end

$LDFLAGS += ' -lpthread' if RUBY_PLATFORM =~ /linux/

# Create Makefile
create_makefile('executorch/executorch')

puts "\nConfiguration complete. Run 'make' to build."
