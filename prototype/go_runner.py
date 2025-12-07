import subprocess
import os
import json
import shutil
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path


def run_go_module(
        project_dir: str,
        #main_file: str = "server/server.go",
        main_file: str = "client_new/client_new.go",
        args: List[str] = None,
        input_data: Optional[str] = None,
        timeout: int = 30,
        env: Optional[Dict[str, str]] = None
) -> Tuple[str, str, int]:
    """
    Run a Go program that has dependencies and a go.mod file.

    Args:
        project_dir: Directory containing go.mod and Go source files
        main_file: Main Go file to run (relative to project_dir)
        args: Optional list of command-line arguments
        input_data: Optional string to pass to stdin
        timeout: Timeout in seconds
        env: Optional environment variables to add/override

    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    project_path = Path(project_dir).resolve()

    if not project_path.exists():
        return "", f"Error: Directory {project_dir} not found", -1

    go_mod_path = project_path / "go.mod"
    if not go_mod_path.exists():
        return "", f"Error: go.mod not found in {project_dir}", -1

    main_file_path = project_path / main_file
    if not main_file_path.exists():
        return "", f"Error: {main_file} not found in {project_dir}", -1

    # Build command
    command = ['go', 'run', str(main_file)]
    if args:
        command.extend(args)

    # Setup environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        # Run from the project directory
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_path),
            env=run_env
        )

        return result.stdout, result.stderr, result.returncode

    except subprocess.TimeoutExpired:
        return "", "Error: Go program timed out", -1
    except FileNotFoundError:
        return "", "Error: Go is not installed or not in PATH", -1
    except Exception as e:
        return "", f"Error: {str(e)}", -1


def setup_go_module(
        project_dir: str,
        module_name: str = None,
        dependencies: List[str] = None
) -> Tuple[bool, str]:
    """
    Initialize a Go module and install dependencies.

    Args:
        project_dir: Directory to create/setup the module
        module_name: Module name (e.g., "example.com/myapp")
        dependencies: List of dependencies to install (e.g., ["github.com/gin-gonic/gin@latest"])

    Returns:
        Tuple of (success, message)
    """
    project_path = Path(project_dir).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    go_mod_path = project_path / "go.mod"

    # Initialize module if go.mod doesn't exist
    if not go_mod_path.exists():
        if not module_name:
            module_name = project_path.name

        try:
            result = subprocess.run(
                ['go', 'mod', 'init', module_name],
                capture_output=True,
                text=True,
                cwd=str(project_path),
                timeout=30
            )

            if result.returncode != 0:
                return False, f"Failed to initialize module: {result.stderr}"

        except Exception as e:
            return False, f"Error initializing module: {str(e)}"

    # Install dependencies
    if dependencies:
        for dep in dependencies:
            try:
                result = subprocess.run(
                    ['go', 'get', dep],
                    capture_output=True,
                    text=True,
                    cwd=str(project_path),
                    timeout=60
                )

                if result.returncode != 0:
                    return False, f"Failed to install {dep}: {result.stderr}"

            except Exception as e:
                return False, f"Error installing {dep}: {str(e)}"

    # Run go mod tidy to clean up
    try:
        result = subprocess.run(
            ['go', 'mod', 'tidy'],
            capture_output=True,
            text=True,
            cwd=str(project_path),
            timeout=30
        )

        if result.returncode != 0:
            return False, f"Failed to tidy module: {result.stderr}"

    except Exception as e:
        return False, f"Error tidying module: {str(e)}"

    return True, "Module setup successful"


def build_go_module(
        project_dir: str,
        output_name: str = None,
        #main_file: str = "server/server.go",
        main_file: str = "client_new/client_new.go",
        build_flags: List[str] = None
) -> Tuple[bool, str, str]:
    """
    Build a Go module into an executable.

    Args:
        project_dir: Directory containing go.mod and Go source files
        output_name: Name of output executable (default: directory name)
        main_file: Main Go file to build
        build_flags: Additional build flags (e.g., ["-ldflags", "-s -w"])

    Returns:
        Tuple of (success, output_path, error_message)
    """
    project_path = Path(project_dir).resolve()

    if not project_path.exists():
        return False, "", f"Directory {project_dir} not found"

    if not output_name:
        output_name = project_path.name

    # Add .exe extension on Windows
    if os.name == 'nt' and not output_name.endswith('.exe'):
        output_name += '.exe'

    output_path = project_path / output_name

    command = ['go', 'build', '-o', str(output_path)]
    if build_flags:
        command.extend(build_flags)
    command.append(main_file)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(project_path),
            timeout=120
        )

        if result.returncode != 0:
            return False, "", f"Build failed: {result.stderr}"

        return True, str(output_path), "Build successful"

    except Exception as e:
        return False, "", f"Error during build: {str(e)}"


def run_built_executable(
        executable_path: str,
        args: List[str] = None,
        input_data: Optional[str] = None,
        timeout: int = 30,
        cwd: Optional[str] = None
) -> Tuple[str, str, int]:
    """
    Run a compiled Go executable.

    Args:
        executable_path: Path to the executable
        args: Optional command-line arguments
        input_data: Optional stdin data
        timeout: Timeout in seconds
        cwd: Working directory (default: executable's directory)

    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    exe_path = Path(executable_path).resolve()

    if not exe_path.exists():
        return "", f"Error: Executable {executable_path} not found", -1

    command = [str(exe_path)]
    if args:
        command.extend(args)

    if cwd is None:
        cwd = str(exe_path.parent)

    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )

        return result.stdout, result.stderr, result.returncode

    except subprocess.TimeoutExpired:
        return "", "Error: Program timed out", -1
    except Exception as e:
        return "", f"Error: {str(e)}", -1


def get_module_info(project_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a Go module.

    Args:
        project_dir: Directory containing go.mod

    Returns:
        Dictionary with module information or None if error
    """
    project_path = Path(project_dir).resolve()

    try:
        # Get module name
        result = subprocess.run(
            ['go', 'list', '-m'],
            capture_output=True,
            text=True,
            cwd=str(project_path),
            timeout=10
        )

        if result.returncode != 0:
            return None

        module_name = result.stdout.strip()

        # Get dependencies
        result = subprocess.run(
            ['go', 'list', '-m', '-json', 'all'],
            capture_output=True,
            text=True,
            cwd=str(project_path),
            timeout=30
        )

        if result.returncode != 0:
            return None

        # Parse JSON output (multiple JSON objects)
        dependencies = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    dep = json.loads(line)
                    if dep.get('Path') != module_name:  # Exclude self
                        dependencies.append({
                            'path': dep.get('Path'),
                            'version': dep.get('Version')
                        })
                except json.JSONDecodeError:
                    pass

        return {
            'module_name': module_name,
            'dependencies': dependencies,
            'dependency_count': len(dependencies)
        }

    except Exception as e:
        print(f"Error getting module info: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example 1: Create a new Go module with dependencies
    print("=== Example 1: Creating Go module with dependencies ===")

    project_dir = "/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG"

    # Setup the module
    success, msg = setup_go_module(
        project_dir,
        module_name="server"
    )

    print(f"Setup: {msg}")

    if success:

        # Run the program
        print("\n=== Running the program ===")
        stdout, stderr, code = run_go_module(
            project_dir,
            #args=["-port","50051"]
            args=["-ip","localhost:50051","-thread","1"]
        )

        print(f"Output:\n{stdout}")
        if stderr:
            print(f"Errors:\n{stderr}")
        print(f"Return code: {code}")

        # Get module info
        print("\n=== Module Information ===")
        info = get_module_info(project_dir)
        if info:
            print(f"Module: {info['module_name']}")
            print(f"Dependencies ({info['dependency_count']}):")
            for dep in info['dependencies'][:5]:  # Show first 5
                print(f"  - {dep['path']} {dep['version']}")

        # Build the executable
        print("\n=== Building executable ===")
        success, exe_path, msg = build_go_module(
            project_dir,
            #output_name="server_exe"
            #output_name="client_exe"
            output_name="batched_client_exe"
        )

        if success:
            print(f"Build successful: {exe_path}")

            # Run the built executable
            print("\n=== Running built executable ===")
            stdout, stderr, code = run_built_executable(
                exe_path,
                args=["-ip","localhost:50051","-thread","1"]
                #args=["-port", "50051"]
            )
            print(f"Output:\n{stderr}")
        else:
            print(f"Build failed: {msg}")

        # Cleanup
        print("\n=== Cleaning up ===")
        #shutil.rmtree(project_dir)
        print("Cleanup complete")
