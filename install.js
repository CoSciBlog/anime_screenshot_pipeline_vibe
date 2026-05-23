module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        message: "git submodule update --init --recursive"
      }
    },
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
          "uv pip install -r requirements.txt",
          "uv pip install -e waifuc",
          "uv pip install -e ."
        ]
      }
    },
    {
      when: "{{gpu !== 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install torch torchvision torchaudio",
          "uv pip install -r requirements.txt",
          "uv pip install -e waifuc",
          "uv pip install -e ."
        ]
      }
    }
  ]
}
