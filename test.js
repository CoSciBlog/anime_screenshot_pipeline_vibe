module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: "python -m pytest -q"
      }
    }
  ]
}
