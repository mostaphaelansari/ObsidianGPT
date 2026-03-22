# ObsidianGPT 🔮

**ObsidianGPT** is an elite, academically rigorous, productivity-maximizing second-brain assistant deeply embedded inside your local Obsidian vault. It acts as an autonomous knowledge synthesizer that strictly relies on the context of your personal notes avoiding hallucinations and ensuring high-signal reasoning.

![ObsidianGPT Enterprise UI Demo](assets/demo.webp)

## 🌟 Key Features
- **Local-First Inference:** Powered by `ollama` (default: `qwen3:1.7b` or switchable to `gemini` via API keys), ensuring your private vault data never leaves your machine unless you explicitly configure it otherwise.
- **Agentic Synthesis Pipeline:** Dynamically plans execution steps, retrieves relevant chunks from your `.md` files, and synthesizes answers using academic-grade reasoning.
- **Enterprise Web Workspace:** A polished, multi-view Single Page Application (SPA) natively integrated with FastAPI, offering:
  - **Research Agent:** Formulate goals and view retrieved evidence.
  - **Session History:** Review past reasoning runs in the current session.
  - **Vault Files:** Live indexing of available context notes.
  - **Agent Config:** On-the-fly tuning of models and chunk limits.
- **Obsidian-Native Output:** Understands and generates perfect Obsidian Markdown (`[[wikilinks]]`, `#tags`, YAML frontmatter, `$math$`).

## 🚀 Installation

ObsidianGPT requires Python 3.10+ and a configured vault.

1. Clone the repository:
```bash
git clone https://github.com/mostaphaelansari/ObsidianGPT-.git
cd ObsidianGPT-
```
2. Install the package and dependencies:
```bash
pip install -e .
```
3. Set your environment variables in a `.env` file! Required variables:
- `VAULT_PATH`="/path/to/your/obsidian/vault"
- `OLLAMA_MODEL`="qwen3:1.7b"
- `GEMINI_API_KEY`="..." (optional, for Gemini fallback)

## 🛠️ Usage
ObsidianGPT offers two powerful usage modes.

### 1. Enterprise Web Workspace
Launch the FastAPI web server to access the GUI:
```powershell
uvicorn obsidiangpt.server:app --port 8000
```
Then navigate to `http://localhost:8000` in your browser.

### 2. Command Line Interface (CLI)
You can directly query your vault from the terminal using the `obsidiangpt` command.
```bash
obsidiangpt "What is the difference between supervised and unsupervised learning?"
```
This will parse the vault, execute the reasoning pipeline, and output a structured `Synthesis.md` file directly into your workspace.

---
**ObsidianGPT** — Function over form. Zero hallucination. High-signal output.
