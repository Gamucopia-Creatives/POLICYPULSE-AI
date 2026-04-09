import sys
import os
# Add parent directory to path so we can import 'envs' and 'inference'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from envs.social_stream_moderation.environment import SocialStreamModerationEnv
from envs.social_stream_moderation.models import State, ModerationAction
from envs.social_stream_moderation.graders import list_graders as _list_graders, get_grader
from envs.social_stream_moderation.tasks import TASKS

# Enums for Swagger Dropdowns
class TaskName(str, Enum):
    TASK_1 = "Task 1: Basic Safety"
    TASK_2 = "Task 2: Context & Nuance"
    TASK_3 = "Task 3: Fairness & Bias"

class PolicyModeChoice(str, Enum):
    NORMAL = "Standard Moderation"
    STRICT = "Strict Enforcement"
    LENIENT = "Lenient Privacy"

class UserHistoryChoice(str, Enum):
    CLEAN = "Clean History"
    REPEATED = "Repeat Offender"

class ContextTypeChoice(str, Enum):
    ROOT = "Main Post"
    COMMENT = "Comment"

# Mapping UI labels back to backend IDs
TASK_MAP = {
    TaskName.TASK_1: "Task 1: Basic Safety",
    TaskName.TASK_2: "Task 2: Context & Nuance",
    TaskName.TASK_3: "Task 3: Fairness & Bias"
}

POLICY_MAP = {
    PolicyModeChoice.NORMAL: "normal",
    PolicyModeChoice.STRICT: "strict",
    PolicyModeChoice.LENIENT: "lenient"
}

HISTORY_MAP = {
    UserHistoryChoice.CLEAN: "no_prior_violations",
    UserHistoryChoice.REPEATED: "prior_violations"
}

CONTEXT_MAP = {
    ContextTypeChoice.ROOT: "root_post",
    ContextTypeChoice.COMMENT: "comment"
}

# API Metadata for Swagger
TAGS_METADATA = [
    {
        "name": "🤖 Automated Benchmarking",
        "description": "Autonomous evaluation loop. Sequence: **Reset** -> **Predict & Step** (Repeat). This tracks the official hackathon metrics.",
    },
    {
        "name": "🧪 Interactive Lab",
        "description": "Manual testing endpoints. Perfect for testing specific edge cases with custom inputs and human overrides.",
    },
    {
        "name": "📊 System Monitoring",
        "description": "Real-time state and status tracking for the moderation engine.",
    }
]

app = FastAPI(
    title="🛡️ PolicyPulse AI | Intelligence Center",
    description="""
### Evaluation Guide for Hackathon Judges:
1. **Automated Testing:** Use `[POST] /reset` then `[POST] /predict_and_step`.
2. **Fairness Testing (Task 3):** Start an episode with `task_name='policy_fairness'`.
3. **Internal Logic:** Use `[POST] /evaluate` to see the model's reasoning without advancing the environment.
    """,
    version="1.2.0",
    openapi_tags=TAGS_METADATA
)
env = SocialStreamModerationEnv()

class ResetRequest(BaseModel):
    task_name: TaskName = Field(TaskName.TASK_1, description="Select the benchmark level to initialize.")
    seed: Optional[int] = Field(42, description="Reproducibility seed for dataset sampling.")

class EvaluateRequest(BaseModel):
    text: str = Field("I will kill you", description="The user content string to analyze.")
    api_base_url: Optional[str] = Field(None, description="Optional override. If blank, defaults to server's API_BASE_URL config.")
    model_name: Optional[str] = Field(None, description="Optional override. If blank, defaults to server's MODEL_NAME config.")
    api_key: Optional[str] = Field(None, description="Optional override. If blank, defaults to server's HF_TOKEN config.")

class LLMConfigRequest(BaseModel):
    api_base_url: Optional[str] = Field(None, description="Optional override. If blank, defaults to server's API_BASE_URL config.")
    model_name: Optional[str] = Field(None, description="Optional override. If blank, defaults to server's MODEL_NAME config.")
    api_key: Optional[str] = Field(None, description="Optional override. If blank, defaults to server's HF_TOKEN config.")

class StepRequest(BaseModel):
    action: ModerationAction = Field(ModerationAction.ALLOW, description="The action to apply to the current post.")

class FeedbackRequest(BaseModel):
    text: str
    corrected_action: ModerationAction
    reason: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    return r"""

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PolicyPulse AI | Intelligence Center</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #030712;
                --sidebar: rgba(15, 23, 42, 0.6);
                --accent: #38bdf8;
                --danger: #f472b6;
                --success: #4ade80;
                --text: #f8fafc;
                --muted: #94a3b8;
            }
            * { margin:0; padding:0; box-sizing:border-box; }
            body {
                font-family:'Outfit', sans-serif; background: #030712; color:var(--text);
                height:100vh; overflow:hidden; display:flex; flex-direction:column;
                transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Custom Scrollbars */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.2); border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

            main { 
                flex:1; 
                display:grid; 
                grid-template-columns: 320px 1fr 0px; 
                gap:20px; 
                padding:20px; 
                max-height:calc(100vh - 60px); 
                transition: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            body.audit-active main { 
                grid-template-columns: 320px 1fr 420px; 
            }

            header { height:60px; display:flex; align-items:center; justify-content:space-between; padding:0 30px; border-bottom:1px solid rgba(255,255,255,0.05); background:rgba(15, 23, 42, 0.4); }
            .logo { font-weight:800; font-size:1.4rem; letter-spacing:-0.03em; color:var(--accent); }
            .version { font-size:0.7rem; background:rgba(56, 189, 248, 0.1); padding:4px 10px; border-radius:6px; color:var(--accent); font-weight:600; }

            /* Panel Styling */
            .panel { background:var(--sidebar); backdrop-filter:blur(20px); border-radius:24px; border:1px solid rgba(255,255,255,0.06); display:flex; flex-direction:column; overflow:hidden; }
            .panel-header { padding:25px; border-bottom:1px solid rgba(255,255,255,0.05); }
            .panel-title { font-size:0.9rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; display:flex; align-items:center; gap:10px; }
            .panel-title::before { content:''; width:3px; height:14px; background:var(--accent); border-radius:10px; }
            .panel-content { padding:25px; flex:1; overflow-y:auto; }

            /* Tabs */
            .mode-switch { display:flex; background:rgba(0,0,0,0.3); padding:4px; border-radius:12px; margin-bottom:25px; }
            .tab { flex:1; padding:10px; text-align:center; cursor:pointer; font-size:0.8rem; font-weight:700; border-radius:8px; transition:0.3s; color:var(--muted); }
            .tab.active { background:var(--accent); color:#020617; }

            /* Forms */
            .field { margin-bottom:20px; }
            label { display:block; font-size:0.65rem; font-weight:700; color:var(--muted); text-transform:uppercase; margin-bottom:8px; }
            select, textarea { width:100%; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:12px; color:#fff; font-family:'Outfit'; font-size:0.9rem; transition:0.3s; }
            textarea { resize:none; min-height:100px; }
            select:focus, textarea:focus { outline:none; border-color:var(--accent); }

            /* Buttons */
            .btn { width:100%; padding:16px; border-radius:14px; border:none; font-weight:700; cursor:pointer; transition:0.3s; font-size:0.95rem; display:flex; align-items:center; justify-content:center; gap:10px; }
            .btn-primary { background:var(--accent); color:#020617; }
            .btn-primary:hover { background:#7dd3fc; transform:translateY(-2px); }
            .btn-secondary { background:rgba(255,255,255,0.05); color:#fff; border:1px solid rgba(255,255,255,0.1); margin-top:10px; }
            .btn-secondary:hover { background:rgba(255,255,255,0.08); }
            .btn:disabled { opacity:0.3; cursor:not-allowed; transform:none !important; }

            /* Right Column */
            .stats-bar { display:grid; grid-template-columns: repeat(3, 1fr); gap:15px; margin-bottom:20px; }
            .stat-card { background:rgba(255,255,255,0.03); padding:15px; border-radius:16px; border:1px solid rgba(255,255,255,0.05); }
            .stat-label { font-size:0.6rem; color:var(--muted); font-weight:700; text-transform:uppercase; }
            .stat-value { font-size:1.1rem; font-weight:800; font-family:'JetBrains Mono'; margin-top:5px; color:var(--accent); }

            .log-container { background:rgba(0,0,0,0.2); border-radius:20px; border:1px solid rgba(255,255,255,0.05); flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
            .log-entry { 
                background:rgba(255,255,255,0.02); padding:18px; border-radius:14px; 
                border-left:3px solid var(--accent); animation:fadeIn 0.3s; 
                transition:0.3s; cursor:default;
            }
            .log-entry.active-audit { background:rgba(56,189,248,0.08); border-color:var(--accent); box-shadow:0 10px 30px rgba(0,0,0,0.3); }
            @keyframes fadeIn { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
            .log-meta { display:flex; justify-content:space-between; font-size:0.7rem; color:var(--muted); margin-bottom:8px; font-weight:600; }
            .log-text { font-size:0.95rem; line-height:1.4; color:#e2e8f0; }
            .log-badge { font-size:0.6rem; font-weight:800; padding:2px 8px; border-radius:4px; text-transform:uppercase; margin-top:10px; display:inline-block; }

            .audit-btn { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); color:var(--muted); font-size:0.6rem; padding:4px 12px; border-radius:6px; cursor:pointer; font-weight:800; transition:0.2s; }
            .audit-btn:hover { background:var(--danger); color:#000; border-color:var(--danger); }
            
            .verify-btn { background:rgba(74,222,128,0.05); border:1px solid var(--success); color:var(--success); font-size:0.6rem; padding:4px 12px; border-radius:6px; cursor:pointer; font-weight:800; transition:0.2s; }
            .verify-btn:hover { background:var(--success); color:#000; }

            .grid-btn { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); color:white; font-size:0.7rem; padding:12px; border-radius:8px; cursor:pointer; font-weight:700; transition:0.2s; }
            .grid-btn:hover { background:var(--accent); color:#020617; border-color:var(--accent); }

            /* Skeleton Shimmer */
            .skeleton { 
                height: 200px; 
                background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%);
                background-size: 200% 100%;
                animation: shimmer 1.5s infinite;
                border-radius: 14px;
                margin-bottom: 12px;
                border: 1px solid rgba(255,255,255,0.05);
            }
            @keyframes shimmer {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }

            .empty-state { margin:auto; text-align:center; color:var(--muted); font-weight:300; }

            /* Header Nav */
            .nav-links { display:flex; gap:25px; align-items:center; }
            .nav-links a { 
                font-size:0.75rem; 
                color:var(--muted); 
                text-decoration:none; 
                font-weight:700; 
                letter-spacing:0.05em; 
                transition:0.3s; 
                position:relative;
                padding-bottom: 4px;
            }
            .nav-links a:hover { color:var(--accent); }
            .nav-links a::after {
                content: '';
                position: absolute;
                bottom: 0; left: 0;
                width: 0; height: 1px;
                background: var(--accent);
                transition: 0.3s;
            }
            .nav-links a:hover::after { width: 100%; }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">POLICYPULSE <span style="font-weight:300">AI</span></div>
            <div style="display:flex; align-items:center; gap:20px;">
                <div class="nav-links">
                    <a href="/docs">API REFERENCE</a>
                    <a href="/state">SYSTEM STATUS</a>
                </div>
                <div class="version">REVISION 1.0</div>
            </div>
        </header>

        <main>
            <!-- Left Panel: Orchestration -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Operation Center</div>
                </div>
                <div class="panel-content">
                    <div class="mode-switch">
                        <div class="tab active" id="tab-lab">LIVE MODE</div>
                        <div class="tab" id="tab-auto">GRADER MODE</div>
                    </div>

                    <!-- Lab Mode Form -->
                    <div id="section-lab">
                        <div class="field">
                            <label>User Content</label>
                            <textarea id="lab-input" placeholder="Type or paste text to test our agent's moderation logic..."></textarea>
                        </div>
                        <div class="field">
                            <label>Safety Policy</label>
                            <select id="lab-policy">
                                <option value="NORMAL">Standard Moderation</option>
                                <option value="STRICT">Strict Enforcement</option>
                                <option value="LENIENT">Lenient Privacy</option>
                            </select>
                        </div>
                        <div class="field" style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                            <div>
                                <label>User History</label>
                                <select id="lab-history" style="font-size:0.75rem;">
                                    <option value="no_prior_violations">Clean History</option>
                                    <option value="prior_violations">Repeat Offender</option>
                                </select>
                            </div>
                            <div>
                                <label>Context Type</label>
                                <select id="lab-context" style="font-size:0.75rem;">
                                    <option value="root_post">Main Post</option>
                                    <option value="comment">Comment</option>
                                </select>
                            </div>
                        </div>
                    </div>


                    <!-- Auto Mode Form -->
                    <div id="section-auto" style="display:none;">
                        <div class="field">
                            <label>Benchmark Level</label>
                            <select id="auto-task">
                                <option value="Task 1: Basic Safety">Task 1: Basic Safety</option>
                                <option value="Task 2: Context & Nuance">Task 2: Context & Nuance</option>
                                <option value="Task 3: Fairness & Bias">Task 3: Fairness & Bias</option>
                            </select>
                        </div>
                        <button class="btn btn-primary" id="btn-auto-reset">START BENCHMARK</button>
                        <button class="btn btn-secondary" id="btn-auto-step" disabled>PROCESS NEXT ITEM</button>
                    </div>

                    <div style="margin-top:20px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.05);">
                        <div style="font-size:0.65rem; font-weight:700; color:var(--accent); text-transform:uppercase; margin-bottom:10px;">Optional: Custom LLM Override</div>
                        <div class="field" style="margin-bottom:15px;">
                            <input type="text" id="config-base-url" placeholder="API Base URL (e.g., https://api.openai.com/v1)" style="width:100%; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px; color:#fff; font-family:'Outfit'; font-size:0.8rem; margin-bottom:8px;">
                            <input type="text" id="config-model" placeholder="Model Name (e.g., gpt-4o-mini)" style="width:100%; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px; color:#fff; font-family:'Outfit'; font-size:0.8rem; margin-bottom:8px;">
                            <input type="password" id="config-key" placeholder="API Key" style="width:100%; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px; color:#fff; font-family:'Outfit'; font-size:0.8rem;">
                        </div>
                    </div>

                    <button class="btn btn-primary" id="btn-lab-run" style="margin-top:20px" disabled>RUN MODERATION</button>
                    <button class="btn btn-secondary" id="btn-global-clear" style="margin-top:10px">PURGE LOGS</button>

                </div>
            </div>

            <!-- Right Panel: Intelligence Stream -->
            <div class="panel" style="background:transparent; border:none; backdrop-filter:none;">
                <div class="stats-bar">
                    <div class="stat-card">
                        <div class="stat-label">Model Accuracy</div>
                        <div class="stat-value" id="val-accuracy">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Aggregate Reward</div>
                        <div class="stat-value" id="val-reward">0.000</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">System State</div>
                        <div class="stat-value" id="val-state" style="color:var(--muted)">IDLE</div>
                    </div>
                </div>

                <div class="log-container" id="log-viewport">
                    <div class="empty-state" id="empty-hint">
                        <div style="font-size:3rem; margin-bottom:20px; opacity:0.2;">📉</div>
                        <div style="font-weight:600; font-size:0.9rem;">Intelligence Stream Idle</div>
                        <p style="font-size:0.75rem; opacity:0.5; margin-top:10px;">Configure your parameters and click 'RUN MODERATION' to begin ingestion.</p>
                    </div>
                </div>
            </div>

            <!-- Audit Inspector Sidepanel (Now correctly part of the grid) -->
            <div id="inspector-pane" class="panel" style="border-left:1px solid rgba(255,255,255,0.1); background:rgba(15,23,42,0.6); overflow:hidden; visibility:hidden; opacity:0; transition:0.4s;">
                <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="panel-title">Audit Inspector</div>
                    <button onclick="closeInspector()" style="background:none; border:none; color:var(--muted); cursor:pointer; font-size:1.2rem;">&times;</button>
                </div>
                <div class="panel-content" id="inspector-content" style="padding:20px;">
                    <!-- Content injected by JS -->
                </div>
            </div>
        </main>

        <script>
            // Elements
            const tabs = { lab: document.getElementById('tab-lab'), auto: document.getElementById('tab-auto') };
            const sections = { lab: document.getElementById('section-lab'), auto: document.getElementById('section-auto') };
            const btnLabRun = document.getElementById('btn-lab-run');
            const btnAutoReset = document.getElementById('btn-auto-reset');
            const btnAutoStep = document.getElementById('btn-auto-step');
            const btnGlobalClear = document.getElementById('btn-global-clear');
            const logViewport = document.getElementById('log-viewport');

            // HUD
            const valReward = document.getElementById('val-reward');
            const valAccuracy = document.getElementById('val-accuracy');
            const valState = document.getElementById('val-state');

            let totalReward = 0;
            let counter = 0;
            let currentMode = 'lab';

            // Tab Switching
            tabs.lab.onclick = () => setMode('lab');
            tabs.auto.onclick = () => setMode('auto');

            // Mode Switch Logic
            function setMode(m) {
                currentMode = m;
                sections.lab.style.display = m === 'lab' ? 'block' : 'none';
                sections.auto.style.display = m === 'auto' ? 'block' : 'none';
                tabs.lab.classList.toggle('active', m === 'lab');
                tabs.auto.classList.toggle('active', m === 'auto');
                
                // Clear state UI
                valState.textContent = 'READY';
                valState.style.color = 'var(--accent)';
            }
            
            // Lab Input Validation
            document.getElementById('lab-input').oninput = (e) => {
                btnLabRun.disabled = !e.target.value.trim();
            };

            // Task Change Re-enables Start
            document.getElementById('auto-task').onchange = () => {
                btnAutoReset.disabled = false;
                btnAutoStep.disabled = true;
            };

            // Global Reset
            btnGlobalClear.onclick = () => {
                logViewport.innerHTML = '<div class="empty-state">System purged. Waiting for new data.</div>';
                totalReward = 0;
                counter = 0;
                valReward.textContent = '0.000';
                valAccuracy.textContent = '--';
                valState.textContent = 'IDLE';
                valState.style.color = 'var(--muted)';
                btnAutoStep.disabled = true;
                btnAutoReset.disabled = false; // Re-enable benchmark
                if (currentMode === 'auto') valState.textContent = 'SYSTEM RESET';
            };

            // Lab Evaluation
            btnLabRun.onclick = async () => {
                const text = document.getElementById('lab-input').value.trim();
                const policy = document.getElementById('lab-policy').value;
                const history = document.getElementById('lab-history').value;
                const context = document.getElementById('lab-context').value;
                if (!text) return;

                btnLabRun.disabled = true;

                // Show Skeleton Loading State
                const skeleton = document.createElement('div');
                skeleton.id = 'lab-shimmer';
                skeleton.innerHTML = `<div class="skeleton"></div>`;
                logViewport.prepend(skeleton);

                try {
                    const resp = await fetch('/evaluate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: text,
                            policy_mode: policy.toLowerCase(),
                            user_history: history,
                            context_type: context,
                            api_base_url: document.getElementById('config-base-url').value.trim() || undefined,
                            model_name: document.getElementById('config-model').value.trim() || undefined,
                            api_key: document.getElementById('config-key').value.trim() || undefined
                        })
                    });

                    if (skeleton) skeleton.remove();

                    if (!resp.ok) {
                        const errData = await resp.json();
                        throw new Error(errData.detail || "Neural Evaluation Failed");
                    }

                    const data = await resp.json();
                    renderEntry(text, data.action, data.reward, policy, data.reason, {history, context});
                    updateHUD(data.reward);
                    document.getElementById('lab-input').value = '';
                    btnLabRun.disabled = true; // Auto-disable after clear
                } catch (e) {
                    if (skeleton) skeleton.remove();
                    const errorEntry = document.createElement('div');
                    errorEntry.className = 'log-entry';
                    errorEntry.style.borderLeftColor = 'var(--danger)';
                    errorEntry.style.background = 'rgba(244,114,182,0.05)';
                    errorEntry.innerHTML = `
                        <div class="log-meta"><span style="color:var(--danger)">⚠️ SYSTEM EXCEPTION</span></div>
                        <div class="log-text">The neural bridge was interrupted. Check your API key or model availability.</div>
                        <div style="font-size:0.6rem; color:var(--danger); margin-top:10px; opacity:0.7; font-family:'JetBrains Mono'">${e.toString()}</div>
                    `;
                    logViewport.prepend(errorEntry);
                } finally {
                    btnLabRun.disabled = !document.getElementById('lab-input').value.trim();
                }
            };


            // Auto Benchmark
            btnAutoReset.onclick = async () => {
                btnAutoReset.disabled = true; // Lock Benchmark
                btnAutoStep.disabled = false;
                const task = document.getElementById('auto-task').value;
                valState.textContent = 'RESETTING...';
                const resp = await fetch('/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task_name: task})
                });
                const state = await resp.json();

                logViewport.innerHTML = `<div class="log-entry" style="border-color:var(--muted)">
                    <div class="log-meta"><span>SYSTEM EVENT</span><span>SESSION START</span></div>
                    <div class="log-text">Environment reset complete. Target: <b>${task}</b>. Dataset contains ${state.total_steps} items. Ready for sequential evaluation.</div>
                </div>`;


                valState.textContent = `SEQ: 1/${state.total_steps}`;
                btnAutoStep.disabled = false;
            };

            btnAutoStep.onclick = async () => {
                if (btnAutoStep.disabled) return;
                btnAutoStep.disabled = true;

                // Show Skeleton Loading State
                const logViewport = document.getElementById('log-viewport');
                const skeleton = document.createElement('div');
                skeleton.id = 'shimmer-loading';
                skeleton.innerHTML = `<div class="skeleton"></div>`;
                logViewport.prepend(skeleton);

                try {
                    const stateResp = await fetch('/state');
                    const state = await stateResp.json();

                    const evalResp = await fetch('/evaluate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: state.text, 
                            policy_mode: state.platform_policy_mode,
                            api_base_url: document.getElementById('config-base-url').value.trim() || undefined,
                            model_name: document.getElementById('config-model').value.trim() || undefined,
                            api_key: document.getElementById('config-key').value.trim() || undefined
                        })
                    });
                    const evalData = await evalResp.json();

                    const stepResp = await fetch('/step', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({action: evalData.action})
                    });
                    const stepResult = await stepResp.json();

                    // Remove Skeleton
                    if (skeleton) skeleton.remove();

                    renderEntry(state.text, evalData.action, stepResult.reward, state.platform_policy_mode.toUpperCase(), evalData.reason, {history: state.user_history_summary, context: state.context_type});
                    updateHUD(stepResult.reward);

                    if (stepResult.done) {
                        valState.textContent = 'COMPLETE';
                        valState.style.color = 'var(--success)';
                        btnAutoStep.disabled = true;

                        logViewport.innerHTML = `<div class="log-entry" style="border-color:var(--success); background:rgba(74,222,128,0.05)">
                            <div class="log-meta"><span>EPISODE COMPLETE</span><span>FINAL GRADE</span></div>
                            <div class="log-text">The environment has finalized this sequence. Total episodes rewards calculated with active fairness parity checks.</div>
                            <div style="font-size:1.4rem; font-weight:800; color:var(--success); margin-top:15px; font-family:'JetBrains Mono'">SCORE: ${stepResult.final_score.toFixed(4)}</div>
                        </div>` + logViewport.innerHTML;
                    } else {
                        valState.textContent = `SEQ: ${state.step_index + 1}/${state.total_steps}`;
                        btnAutoStep.disabled = false;
                    }
                } catch (e) {
                    if (skeleton) skeleton.remove();
                    btnAutoStep.disabled = false;
                    
                    const errorEntry = document.createElement('div');
                    errorEntry.className = 'log-entry';
                    errorEntry.style.borderLeftColor = 'var(--danger)';
                    errorEntry.style.background = 'rgba(244,114,182,0.05)';
                    errorEntry.innerHTML = `
                        <div class="log-meta"><span style="color:var(--danger)">⚠️ SYSTEM EXCEPTION</span></div>
                        <div class="log-text">An intelligence bypass occurred or the connection was interrupted. Please check your Operation Center configuration or API availability.</div>
                        <div style="font-size:0.6rem; color:var(--danger); margin-top:10px; opacity:0.7; font-family:'JetBrains Mono'">${e.toString()}</div>
                    `;
                    logViewport.prepend(errorEntry);
                }
            };

            function updateHUD(r) {
                totalReward += r;
                counter++;
                valReward.textContent = totalReward.toFixed(3);
                valAccuracy.textContent = (totalReward / counter).toFixed(3);
            }

            function renderEntry(text, action, reward, mode, reason, meta) {
                const colors = { ALLOW:'var(--accent)', BAN_USER:'var(--danger)', HARD_FILTER:'var(--danger)', SOFT_HIDE:'#fbbf24', ALLOW_WITH_WARNING:'var(--accent)', ESCALATE_HUMAN:'var(--success)' };
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.style.borderColor = colors[action] || 'var(--accent)';
                entry.innerHTML = `
                    <div class="log-meta">
                        <span>POLICY: ${mode}</span>
                        <span>VERDICT: +${reward.toFixed(3)}</span>
                    </div>
                    <div style="display:flex; gap:8px; margin-bottom:10px;">
                        <span style="font-size:0.6rem; color:var(--muted); border:1px solid rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; text-transform:uppercase;">${meta.history.replace(/_/g,' ')}</span>
                        <span style="font-size:0.6rem; color:var(--muted); border:1px solid rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; text-transform:uppercase;">${meta.context.replace(/_/g,' ')}</span>
                    </div>
                    <div class="log-text">${text}</div>
                    <div style="font-size:0.75rem; color:var(--accent); background:rgba(56,189,248,0.04); padding:12px; border-radius:12px; margin-top:12px; border:1px solid rgba(56,189,248,0.1); white-space: pre-wrap; line-height: 1.6;">
                        ${reason}
                    </div>
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:12px;">
                        <span class="log-badge" style="background:${colors[action] || 'var(--accent)'}; color:#020617; margin-top:0">${action}</span>
                        <div class="hitl-actions" id="hitl-${counter}" style="display:flex; gap:5px;">
                            <button onclick="showOverrideMenu(this, ${reward}, '${action}', \`${text.replace(/`/g, '\\`')}\`)" class="audit-btn">AUDIT</button>
                            <button onclick="verifyAction(this)" class="verify-btn">VERIFY</button>
                        </div>
                    </div>
                `;
                const hint = document.getElementById('empty-hint');
                if (hint) hint.remove();
                logViewport.prepend(entry);
            }

            function verifyAction(btn) {
                btn.parentElement.innerHTML = '<span style="color:var(--success); font-size:0.6rem; font-weight:800; border:1px solid var(--success); padding:2px 6px; border-radius:4px;">✓ HUMAN VERIFIED</span>';
            }

            function closeInspector() {
                document.body.classList.remove('audit-active');
                const pane = document.getElementById('inspector-pane');
                pane.style.visibility = 'hidden';
                pane.style.opacity = '0';
                if (window.__active_row) window.__active_row.classList.remove('active-audit');
            }

            function showOverrideMenu(btn, originalReward, originalAction, originalText) {
                const pane = document.getElementById('inspector-pane');
                const content = document.getElementById('inspector-content');
                const row = btn.closest('.log-entry');
                
                if (window.__active_row) window.__active_row.classList.remove('active-audit');
                row.classList.add('active-audit');
                window.__active_row = row;

                window.__pending_text = originalText;
                window.__pending_reward = originalReward;
                window.__pending_hitl_id = btn.parentElement.id;
                window.__selected_action = null; 
                
                content.innerHTML = `
                    <div style="display:flex; flex-direction:column; gap:20px;">
                        <div style="background:rgba(255,255,255,0.03); padding:20px; border-radius:16px; border:1px solid rgba(255,255,255,0.05);">
                            <div style="font-size:0.6rem; color:var(--muted); text-transform:uppercase; font-weight:800; margin-bottom:10px;">Original Content</div>
                            <div style="font-size:0.9rem; line-height:1.5;">"${originalText}"</div>
                        </div>

                        <div style="display:flex; flex-direction:column; gap:12px;">
                            <label style="font-size:0.65rem; color:var(--danger); font-weight:800; text-transform:uppercase;">Correction Verdict</label>
                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;" id="action-selector">
                                <button onclick="selectAction(this, 'ALLOW')" class="grid-btn">ALLOW</button>
                                <button onclick="selectAction(this, 'ALLOW_WITH_WARNING')" class="grid-btn">WARNING</button>
                                <button onclick="selectAction(this, 'SOFT_HIDE')" class="grid-btn">HIDE</button>
                                <button onclick="selectAction(this, 'ESCALATE_HUMAN')" class="grid-btn">ESCALATE</button>
                                <button onclick="selectAction(this, 'BAN_USER')" class="grid-btn" style="grid-column: span 2;">BAN USER</button>
                            </div>
                        </div>
                        
                        <div style="display:flex; flex-direction:column; gap:10px;">
                            <label style="font-size:0.65rem; color:var(--muted); font-weight:800; text-transform:uppercase;">Memory Reason (Optional)</label>
                            <textarea id="feedback-reason" placeholder="Why is this correction necessary?" style="min-height:100px; font-size:0.85rem; background:rgba(0,0,0,0.4); padding:15px; border:1px solid rgba(255,255,255,0.1); border-radius:12px; color:white; width:100%; resize:none;"></textarea>
                        </div>
                        
                        <button id="btn-submit-feedback" onclick="submitFeedback()" class="btn btn-primary" style="margin-top:10px; opacity:0.5;" disabled>REINFORCE SYSTEM</button>
                        
                        <div id="feedback-status" style="font-size:0.7rem; color:var(--muted); text-align:center;">Select an action to enable submission.</div>
                    </div>
                `;
                
                pane.style.visibility = 'visible';
                pane.style.opacity = '1';
                document.body.classList.add('audit-active');
            }

            function selectAction(btn, action) {
                // Clear state
                const btns = document.querySelectorAll('#action-selector .grid-btn');
                btns.forEach(b => {
                    b.style.background = 'rgba(255,255,255,0.05)';
                    b.style.color = 'white';
                });
                
                // Set active
                btn.style.background = 'var(--accent)';
                btn.style.color = '#020617';
                window.__selected_action = action;
                
                // Enable submit
                const submit = document.getElementById('btn-submit-feedback');
                submit.disabled = false;
                submit.style.opacity = '1';
                document.getElementById('feedback-status').innerHTML = "Ready to reinforce local memory.";
            }

            async function submitFeedback() {
                const action = window.__selected_action;
                const reason = document.getElementById('feedback-reason').value.trim() || "Manual correction by human auditor.";
                const text = window.__pending_text;
                const originalReward = window.__pending_reward;
                const hitlId = window.__pending_hitl_id;
                
                const statusDiv = document.getElementById('feedback-status');
                const submitBtn = document.getElementById('btn-submit-feedback');
                
                submitBtn.disabled = true;
                statusDiv.innerHTML = "⏳ REINFORCING LOGIC...";

                try {
                    await fetch('/feedback', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: text,
                            corrected_action: action,
                            reason: reason
                        })
                    });

                    const correction = - (originalReward + 1.0);
                    updateHUD(correction);

                    const container = document.getElementById(hitlId);
                    container.innerHTML = `<span style="color:var(--danger); font-size:0.6rem; font-weight:800; border:1px solid var(--danger); padding:2px 6px; border-radius:4px;">🧠 MEMORY REINFORCED</span>`;
                    
                    statusDiv.innerHTML = "✅ SYSTEM REINFORCED!";
                    setTimeout(closeInspector, 1000);
                } catch (e) {
                    statusDiv.innerHTML = "❌ MEMORY WRITE FAILED";
                    submitBtn.disabled = false;
                }
            }


        </script>
    </body>
    </html>
    """



@app.post("/reset", tags=["🤖 Automated Benchmarking"], summary="1. Initialize Environment (Task Selection)")
async def reset_env(req: ResetRequest = Body(default=ResetRequest())):
    """Resets the environment with a given task and seed. This must be the first step in any benchmarking track."""
    try:
        # Access the enum value (the human-readable name)
        internal_task_name = TASK_MAP[req.task_name]
        state = await env.reset(task_name=internal_task_name, seed=req.seed)
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks", tags=["🤖 Automated Benchmarking"])
async def list_tasks():
    """Returns the list of tasks available in the environment for discovery."""
    return [
        {
            "id": task_cfg.name,
            "difficulty": task_cfg.difficulty,
            "description": f"Episode length: {task_cfg.episode_length} posts. Policy mode: {task_cfg.policy_mode.value}.",
            "grader_id": task_cfg.grader_id,
        }
        for task_cfg in TASKS.values()
    ]

@app.get("/graders", tags=["🛡️ Automated Benchmarking"])
async def list_graders_endpoint():
    """Returns the list of graders available in the environment for discovery."""
    return _list_graders()

@app.post("/evaluate", tags=["🧪 Interactive Lab"], summary="Test Model Logic (XAI Insight)")
async def evaluate_text(
    req: EvaluateRequest,
    policy_mode: PolicyModeChoice = Query(PolicyModeChoice.NORMAL, description="Select the active safety policy regime."),
    user_history: UserHistoryChoice = Query(UserHistoryChoice.CLEAN, description="Select the author's moderation history."),
    context_type: ContextTypeChoice = Query(ContextTypeChoice.ROOT, description="Select the content hierarchical context.")
):
    """Internal endpoint for the Interactive Sandbox UI. Returns logic reasoning and calculated rewards."""
    # Use proper Enum types
    from envs.social_stream_moderation.models import PolicyMode, State
    from envs.social_stream_moderation.graders import compute_per_post_reward, REWARD_TABLE
    from inference import get_agent

    # Map choice to model enum
    try:
        p_mode = PolicyMode(POLICY_MAP[policy_mode])
    except ValueError:
        p_mode = PolicyMode.NORMAL

    # Predict action using rule-based or LLM logic
    agent = get_agent(api_base_url=req.api_base_url, model_name=req.model_name, api_key=req.api_key)
    mock_state = State(
        post_id="playground_test",
        text=req.text,
        user_history_summary=HISTORY_MAP[user_history],
        context_type=CONTEXT_MAP[context_type],
        platform_policy_mode=p_mode.value,
        user_group="A",
        step_index=0,
        total_steps=1
    )


    action, reason = agent.predict(mock_state)

    # We need to simulate a 'correct' label for reward calculation in demo
    # For simplicity in demo, if the agent bans, we assume it caught something
    # (HarmLabel mapping is internal, so we'll just derive a representative reward)

    # Check if this input matches our known harmful patterns to determine reward
    from envs.social_stream_moderation.models import HarmLabel
    from inference import SAFETY_KEYWORDS
    best_harm_guess = HarmLabel.SAFE
    
    for category, keywords in SAFETY_KEYWORDS.items():
        if any(kw in req.text.lower() for kw in keywords):
            best_harm_guess = category
            break

    reward = compute_per_post_reward(best_harm_guess, action, p_mode)

    return {
        "action": action.value,
        "reward": float(reward),
        "reason": reason
    }


@app.post("/step", tags=["🧪 Interactive Lab"])
async def step_env(req: StepRequest):
    try:
        next_state, reward, done, info = await env.step(req.action)

        final_score = 0.0
        grader_id = None
        if done:
            # The environment now uses the task-specific grader internally;
            # the final score and grader_id are returned in ``info``.
            final_score = info.get("score", 0.0)
            grader_id = info.get("grader_id")

        return {
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "info": info,
            "final_score": final_score,
            "grader_id": grader_id,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_and_step", tags=["🤖 Automated Benchmarking"], summary="2. Autonomous Model Execution (Autonomous)")
async def predict_and_step(req: Optional[LLMConfigRequest] = Body(None)):
    """Predicts using dynamic agent and steps the env automatically. This matches our inference.py autonomous loop."""
    from inference import get_agent

    state = env._get_state()
    if state is None:
        raise HTTPException(status_code=400, detail="No active episode. Please call /reset first.")

    agent = get_agent(
        api_base_url=req.api_base_url if req else None,
        model_name=req.model_name if req else None,
        api_key=req.api_key if req else None
    )
    action, reason = agent.predict(state)

    # Execute the step with the model's prediction
    next_state, reward, done, info = await env.step(action)

    final_score = 0.0
    grader_id = None
    if done:
        # The environment now uses the task-specific grader internally
        final_score = info.get("score", 0.0)
        grader_id = info.get("grader_id")

    return {
        "prediction": action.value,
        "reason": reason,
        "reward": reward,
        "done": done,
        "final_score": final_score,
        "grader_id": grader_id,
        "next_state": next_state,
        "info": info
    }

@app.post("/feedback")
async def save_feedback(req: FeedbackRequest):
    """Saves human correction to local JSON memory for reinforcement learning."""
    import json
    memory_path = os.path.join(os.path.dirname(__file__), "..", "envs", "social_stream_moderation", "human_memory.json")
    
    # Load existing memory
    memory = []
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            try:
                memory = json.load(f)
            except:
                memory = []
                
    # Check for duplicates or update
    found = False
    for entry in memory:
        if entry["text"] == req.text:
            entry["action"] = req.corrected_action
            entry["reason"] = req.reason
            found = True
            break
            
    if not found:
        memory.append({
            "text": req.text,
            "action": req.corrected_action,
            "reason": req.reason
        })
        
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)
        
    return {"status": "success", "message": "Memory reinforced."}

@app.get("/state", tags=["📊 System Monitoring"])
def get_state():
    state = env._get_state()
    if state is None:
        return {
            "status": "Ready",
            "message": "Environment is initialized but no episode is currently active.",
            "how_to_start": "Call 'POST /reset' with a task_name (e.g., 'clear_cut_moderation') to begin benchmarking."
        }
    return state


def kill_port(port):
    import subprocess
    import os
    import sys
    try:
        if sys.platform == "win32":
            # Windows logic
            output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            for line in output.strip().split('\n'):
                if 'LISTENING' in line:
                    pid = line.strip().split()[-1]
                    if pid != str(os.getpid()):
                        print(f"Cleanup: Stopping existing process {pid} on port {port}...")
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
        else:
            # Unix/Mac/Linux logic
            try:
                # Use lsof to find the PID
                output = subprocess.check_output(['lsof', '-ti', f':{port}']).decode().strip()
                if output:
                    for pid in output.split('\n'):
                        if pid != str(os.getpid()):
                            print(f"Cleanup: Stopping existing process {pid} on port {port}...")
                            subprocess.run(['kill', '-9', pid], capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to fuser if lsof is missing
                try:
                    subprocess.run(['fuser', '-k', f'{port}/tcp'], capture_output=True)
                except Exception:
                    pass
    except Exception:
        pass

def main():
    import uvicorn
    # Automatically clear the port before starting to avoid [WinError 10048]
    kill_port(7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

