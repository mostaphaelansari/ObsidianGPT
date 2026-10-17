document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('agent-form');
    const input = document.getElementById('goal-input');
    const generateBtn = document.getElementById('generate-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    
    const subtasksList = document.getElementById('subtasks-list');
    const chunksList = document.getElementById('chunks-list');
    const chunkCount = document.getElementById('chunk-count');
    const markdownOutput = document.getElementById('markdown-output');
    
    // View Routing setup
    const navItems = document.querySelectorAll('.nav-item[data-view]');
    const viewPanels = document.querySelectorAll('.view-panel');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            const targetId = item.getAttribute('data-view');
            viewPanels.forEach(panel => {
                if(panel.id === targetId) {
                    panel.classList.remove('hidden');
                } else {
                    panel.classList.add('hidden');
                }
            });
            
            if(targetId === 'view-history') loadHistory();
            if(targetId === 'view-vault') loadVaultFiles();
            if(targetId === 'view-settings') loadSettings();
        });
    });

    // Configuration Settings Logic
    const settingsForm = document.getElementById('settings-form');
    const settingsModel = document.getElementById('settings-model');
    const settingsTopK = document.getElementById('settings-topk');
    const settingsToast = document.getElementById('settings-toast');

    async function loadSettings() {
        try {
            const res = await fetch('/api/settings');
            const data = await res.json();
            settingsModel.value = data.model || '';
            settingsModel.placeholder = `${data.default_model} (${data.provider} default)`;
            settingsTopK.value = data.top_k_chunks || 5;
            settingsToast.classList.add('hidden');
        } catch (e) {
            console.error("Failed to load settings", e);
        }
    }

    if (settingsForm) {
        settingsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const updates = {
                model: settingsModel.value.trim(),
                top_k_chunks: parseInt(settingsTopK.value, 10)
            };
            try {
                const res = await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updates)
                });
                if(res.ok) {
                    settingsToast.classList.remove('hidden');
                    setTimeout(() => settingsToast.classList.add('hidden'), 3000);
                }
            } catch (e) {
                console.error("Failed to save settings", e);
            }
        });
    }

    // Vault Files Logic
    const vaultList = document.getElementById('vault-list');
    const vaultActivePath = document.getElementById('vault-active-path');
    const vaultPathInput = document.getElementById('vault-path-input');
    const vaultConnectForm = document.getElementById('vault-connect-form');
    const vaultConnectStatus = document.getElementById('vault-connect-status');
    const vaultBrowseBtn = document.getElementById('vault-browse-btn');

    function renderVaultFiles(files) {
        vaultList.innerHTML = '';
        if (!files || files.length === 0) {
            vaultList.innerHTML = '<li class="text-muted">No markdown files found in this vault.</li>';
            return;
        }
        files.forEach(f => {
            const li = document.createElement('li');
            li.style.padding = '8px 0';
            li.style.borderBottom = '1px solid var(--border-subtle)';
            li.innerHTML = `<span class="material-icons-round text-muted" style="font-size:16px; align-middle; margin-right:8px;">description</span> ${f}`;
            vaultList.appendChild(li);
        });
    }

    async function loadVaultFiles() {
        try {
            vaultList.innerHTML = '<li class="text-muted">Loading...</li>';
            const res = await fetch('/api/vault');
            const data = await res.json();
            vaultActivePath.textContent = data.path || 'Not connected';
            if (!vaultPathInput.value) vaultPathInput.value = data.path || '';
            renderVaultFiles(data.files);
        } catch (e) {
            console.error(e);
            vaultList.innerHTML = '<li class="text-danger">Failed to load vault files.</li>';
        }
    }

    if (vaultBrowseBtn) {
        vaultBrowseBtn.addEventListener('click', async () => {
            vaultBrowseBtn.disabled = true;
            vaultConnectStatus.className = 'mt-16 text-muted';
            vaultConnectStatus.textContent = 'Opening folder picker on the server...';
            try {
                const res = await fetch('/api/vault/browse', { method: 'POST' });
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.detail || 'Folder picker unavailable');
                }
                if (!data.path) {
                    vaultConnectStatus.textContent = 'No folder selected.';
                    return;
                }

                vaultPathInput.value = data.path;
                vaultConnectStatus.textContent = 'Folder selected — click Connect to use it.';
            } catch (err) {
                vaultConnectStatus.style.color = 'var(--danger-text, #c0392b)';
                vaultConnectStatus.textContent = `Error: ${err.message}`;
            } finally {
                vaultBrowseBtn.disabled = false;
            }
        });
    }

    if (vaultConnectForm) {
        vaultConnectForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const path = vaultPathInput.value.trim();
            if (!path) return;

            vaultConnectStatus.className = 'mt-16 text-muted';
            vaultConnectStatus.textContent = 'Connecting...';

            try {
                const res = await fetch('/api/vault/connect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: path })
                });
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.detail || 'Failed to connect to vault');
                }

                vaultActivePath.textContent = data.path;
                renderVaultFiles(data.files);

                const warning = data.is_obsidian_vault
                    ? ''
                    : ' (no .obsidian folder found here — double check this is your vault root)';
                vaultConnectStatus.className = 'mt-16';
                vaultConnectStatus.style.color = 'var(--success-text)';
                vaultConnectStatus.textContent = `Connected — ${data.file_count} note(s) found${warning}`;
            } catch (err) {
                vaultConnectStatus.className = 'mt-16';
                vaultConnectStatus.style.color = 'var(--danger-text, #c0392b)';
                vaultConnectStatus.textContent = `Error: ${err.message}`;
            }
        });
    }

    // History Logic
    const historyContainer = document.getElementById('history-container');
    async function loadHistory() {
        try {
            historyContainer.innerHTML = '<p class="text-muted">Loading...</p>';
            const res = await fetch('/api/history');
            const data = await res.json();
            historyContainer.innerHTML = '';
            if(!data.history || data.history.length === 0) {
                historyContainer.innerHTML = '<p class="text-muted">No session history yet. Run a synthesis to see it here!</p>';
                return;
            }
            data.history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'ui-card mb-16';
                div.style.marginBottom = '16px';
                div.innerHTML = `
                    <div class="card-header"><h3 style="margin:0; font-size:15px; font-weight:600;">Goal: ${item.goal}</h3></div>
                    <div class="card-body">
                        <div class="markdown-body" style="font-size:13px; color:var(--text-secondary);">${marked.parse(item.markdown)}</div>
                    </div>
                `;
                historyContainer.appendChild(div);
            });
        } catch (e) {
            console.error(e);
            historyContainer.innerHTML = '<p class="text-danger">Failed to load history.</p>';
        }
    }

    // Configure marked.js to handle Obsidian style [[wikilinks]] slightly better
    // A simple renderer override or frontend regex replace is usually needed for perfect wikilinks,
    // but default marked.js handles the rest perfectly.

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const goal = input.value.trim();
        if (!goal) return;

        // UI Reset & Loading State
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');
        generateBtn.disabled = true;
        
        // Let's add dynamic loading text to show action
        const loadingTexts = ["Planning subtasks...", "Scanning vault...", "Reasoning over evidence...", "Synthesizing answer..."];
        let textIdx = 0;
        const textInterval = setInterval(() => {
            textIdx = (textIdx + 1) % loadingTexts.length;
            document.getElementById('loading-text').innerText = loadingTexts[textIdx];
        }, 3000);

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ goal: goal })
            });

            clearInterval(textInterval);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error occurred');
            }

            const data = await response.json();
            
            // 1. Render Subtasks
            subtasksList.innerHTML = '';
            data.subtasks.forEach(task => {
                const li = document.createElement('li');
                li.textContent = task;
                subtasksList.appendChild(li);
            });

            // 2. Render Retrieved Chunks
            chunkCount.textContent = data.chunks.length;
            chunksList.innerHTML = '';
            data.chunks.forEach(chunk => {
                const div = document.createElement('div');
                div.className = 'chunk-card';
                // textContent, not innerHTML: excerpts are raw note text and may contain HTML
                const source = document.createElement('span');
                source.className = 'chunk-source';
                source.textContent = chunk.path;
                const text = document.createElement('p');
                text.className = 'chunk-text';
                text.textContent = chunk.excerpt;
                div.append(source, text);
                chunksList.appendChild(div);
            });

            // 3. Render Synthesis (Markdown -> HTML)
            // Replace [[Wikilink]] with a stylized anchor tag instead of bold text
            let rawMarkdown = data.markdown_body;
            rawMarkdown = rawMarkdown.replace(/\[\[(.*?)\]\]/g, '<a href="#" class="wikilink">$1</a>');
            
            markdownOutput.innerHTML = marked.parse(rawMarkdown);
            
            // Render LaTeX formulas using KaTeX
            if (window.renderMathInElement) {
                renderMathInElement(markdownOutput, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\(', right: '\\)', display: false},
                        {left: '\\[', right: '\\]', display: true}
                    ],
                    throwOnError: false
                });
            }

            // Show results
            loadingState.classList.add('hidden');
            resultsSection.classList.remove('hidden');

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            clearInterval(textInterval);
            loadingState.classList.add('hidden');
            alert(`Error: ${error.message}`);
        } finally {
            generateBtn.disabled = false;
            document.getElementById('loading-text').innerText = "Agent is thinking...";
        }
    });
});
