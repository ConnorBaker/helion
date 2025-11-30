{
  inputs = {
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };
    nixpkgs.url = "github:nixos/nixpkgs";
    cuda-legacy.url = "github:nixos-cuda/cuda-legacy";
    git-hooks-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/git-hooks.nix";
    };
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  outputs =
    inputs:
    let
      inherit (inputs.flake-parts.lib) mkFlake;
      inherit (inputs.nixpkgs) lib;
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];
      mkNixpkgs =
        system:
        import inputs.nixpkgs {
          # TODO: Due to the way Nixpkgs is built in stages, the config attribute set is not re-evaluated.
          # This is problematic for us because we use it to signal the CUDA capabilities to the overlay.
          # The only way I've found to combat this is to use pkgs.extend, which is not ideal.
          # TODO: This also means that Nixpkgs needs to be imported *with* the correct config attribute set
          # from the start, unless they're willing to re-import Nixpkgs with the correct config.
          config =
            { pkgs }:
            {
              allowUnfree = true;
              cudaCapabilities = [ "8.9" ];
              cudaSupport = true;
            };
          localSystem = { inherit system; };
        };
      # Memoization through lambda lifting.
      nixpkgsInstances = lib.genAttrs systems mkNixpkgs;
    in
    mkFlake { inherit inputs; } {
      inherit systems;

      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];

      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = nixpkgsInstances.${system};

          devShells.default = pkgs.mkShell {
            packages = [
              # pkgs.claude-code
              (pkgs.python3.withPackages (
                ps: with ps; [
                  cmaes
                  scikit-learn
                  scipy

                  filecheck
                  optuna
                  psutil
                  rich
                  pytest
                  torch
                  tqdm
                  triton
                  typing-extensions
                ]
              ))
            ];
          };

          legacyPackages = pkgs;

          pre-commit.settings.hooks = {
            # Formatter checks
            treefmt = {
              enable = true;
              package = config.treefmt.build.wrapper;
            };

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              # Nix
              nixfmt.enable = true;

              # Shell
              shellcheck.enable = true;
              shfmt.enable = true;
            };
          };
        };
    };
}
