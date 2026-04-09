{
  description = "Python development template with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: let
    systems = [
      "x86_64-linux"
    ];
    forEachSystem = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forEachSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      default = pkgs.mkShell {
        packages = with pkgs; [
          uv
        ];

        shellHook = ''
          if [ ! -d ".venv" ]; then
            echo "Creating virtual environment with uv..."
            uv venv .venv
          fi

          if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
          fi

          if [ -f "pyproject.toml" ]; then
            echo "Syncing dependencies with uv..."
            uv sync
          fi

          echo "Python $(python --version 2>/dev/null || true) | uv $(uv --version)"
        '';
      };
    });
  };
}
