// Global State
let allPlayers = [];
let draftedPlayers = new Set(); // Set of player_name
let rosters = {}; // Map of team_name -> [player_dicts]
let draftLog = []; // Array of {player_name, team_name}

const LEAGUE_TEAMS = [
    'Jonathan Taylor Day', 'Hope Mahomes Still Standing', 'Lizard lizard lizard',
    'MR. SNIFFLES', 'Why did I trade JSN?', 'No Mo Toe Joe',
    'In My Football Era', 'Kittle Me This', 'Rippin Darts', 'Erica Loves Sports'
];

// Initialize Rosters
LEAGUE_TEAMS.forEach(team => {
    rosters[team] = [];
});

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeOlineSliders();
    fetchPlayers();
    setupDraftEvents();
    setupExportEvents();
    setupSimButton();
});

// 1. Navigation Tabs
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const contentId = `${tab.dataset.tab}-tab`;
            document.getElementById(contentId).classList.add('active');
        });
    });
}

// 2. O-Line Sliders
function initializeOlineSliders() {
    const sliders = document.querySelectorAll('.oline-slider');
    sliders.forEach(slider => {
        slider.addEventListener('input', () => {
            const team = slider.dataset.team;
            const val = parseFloat(slider.value);
            const display = document.getElementById(`val-${team.toLowerCase()}`);
            
            if (val > 0) {
                display.textContent = `Boosted (+${val.toFixed(1)})`;
                display.className = 'oline-val-display boosted';
            } else if (val < 0) {
                display.textContent = `Downgraded (${val.toFixed(1)})`;
                display.className = 'oline-val-display downgraded';
            } else {
                display.textContent = 'Normal (0.0)';
                display.className = 'oline-val-display';
            }
        });
    });

    // Reset button
    document.getElementById('reset-oline-btn').addEventListener('click', (e) => {
        e.preventDefault();
        sliders.forEach(slider => {
            const team = slider.dataset.team;
            if (team === 'DET') slider.value = -0.5;
            else if (team === 'MIA') slider.value = -1.0;
            else if (team === 'SEA') slider.value = 1.0;
            else slider.value = 0.0;
            slider.dispatchEvent(new Event('input'));
        });
    });

    // Recalculate button
    document.getElementById('recalc-btn').addEventListener('click', (e) => {
        e.preventDefault();
        recalculateRankings();
    });
}

// 3. Data Fetching & Rendering
function showLoading(text = "Loading...") {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-state').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-state').classList.add('hidden');
}

async function fetchPlayers() {
    showLoading("Generating 2026 Draft Rankings...");
    try {
        const response = await fetch('/api/players');
        const res = await response.json();
        if (res.status === 'success') {
            allPlayers = res.players;
            renderCheatSheet();
            populateDraftDropdowns();
            renderRosters();
        } else {
            alert(`Error: ${res.message}`);
        }
    } catch (err) {
        console.error(err);
        alert("Failed to load player data from server.");
    } finally {
        hideLoading();
    }
}

async function recalculateRankings() {
    showLoading("Running PyTorch rankings recalculations...");
    const adjustments = {};
    document.querySelectorAll('.oline-slider').forEach(slider => {
        adjustments[slider.dataset.team] = parseFloat(slider.value);
    });

    try {
        const response = await fetch('/api/recalculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ adjustments })
        });
        const res = await response.json();
        if (res.status === 'success') {
            // Keep drafted players state but update player features & ranks
            allPlayers = res.players;
            renderCheatSheet();
            populateDraftDropdowns();
            // Re-render rosters with updated projected ADPs
            renderRosters();
            // Automatically run playoff odds simulation if draft has started
            if (draftedPlayers.size > 0) {
                runPlayoffSimulation();
            }
        } else {
            alert(`Recalculate failed: ${res.message}`);
        }
    } catch (err) {
        console.error(err);
        alert("Server error during recalculation.");
    } finally {
        hideLoading();
    }
}

function renderCheatSheet() {
    const tbody = document.getElementById('cheat-sheet-tbody');
    tbody.innerHTML = '';
    
    const searchVal = document.getElementById('search-input').value.toLowerCase();
    const activePos = document.querySelector('.filter-btn.active').dataset.pos;
    
    const filtered = allPlayers.filter(p => {
        const matchesSearch = p.player_name.toLowerCase().includes(searchVal) || 
                              p.team_predict.toLowerCase().includes(searchVal);
        const matchesPos = activePos === 'ALL' || (activePos === 'SEA' ? p.team_predict === 'SEA' : p.position === activePos);
        return matchesSearch && matchesPos;
    });

    filtered.forEach(p => {
        const tr = document.createElement('tr');
        const escapedName = p.player_name.replace(/'/g, "\\'");
        if (draftedPlayers.has(p.player_name)) {
            tr.classList.add('drafted');
        }

        const isOlineBoosted = p.oline_score > 0.3;
        const isOlineDowngraded = p.oline_score < -0.3;
        let olineClass = '';
        if (isOlineBoosted) olineClass = 'boosted';
        if (isOlineDowngraded) olineClass = 'downgraded';

        const vorpClass = p.vorp > 0 ? 'oline-val-display boosted' : 'oline-val-display downgraded';
        const vorpText = p.vorp > 0 ? `+${p.vorp.toFixed(1)}` : p.vorp.toFixed(1);

        tr.innerHTML = `
            <td><strong>${p.draft_rank}</strong></td>
            <td><strong>${p.player_name}</strong></td>
            <td><span class="badge pos-${p.position}">${p.position}</span></td>
            <td>${p.team_predict}</td>
            <td class="text-right">${p.predicted_adp.toFixed(2)}</td>
            <td class="text-right">${p.predicted_pts.toFixed(1)}</td>
            <td class="text-right"><span class="${vorpClass}">${vorpText}</span></td>
            <td class="text-right">${p.prev_fantasy_points.toFixed(1)}</td>
            <td class="text-right"><span class="oline-val-display ${olineClass}">${p.oline_score.toFixed(2)}</span></td>
            <td class="text-center" style="position: relative;">
                ${draftedPlayers.has(p.player_name) 
                    ? `<button class="btn btn-secondary btn-small" onclick="undraftPlayer('${escapedName}')"><i class="fa-solid fa-rotate-left"></i> Undo</button>`
                    : `<div class="custom-dropdown">
                           <button class="btn btn-primary btn-small dropdown-trigger" onclick="toggleDropdown(event, '${escapedName}')">
                               Draft to... <i class="fa-solid fa-chevron-down" style="font-size: 9px; margin-left: 2px;"></i>
                           </button>
                           <div class="dropdown-menu hidden" id="dropdown-${p.player_name.replace(/[^a-zA-Z0-9]/g, '-')}">
                               ${LEAGUE_TEAMS.map(team => `
                                   <div class="dropdown-item" onclick="executeDraft('${escapedName}', '${team.replace(/'/g, "\\'")}')">${team}</div>
                               `).join('')}
                           </div>
                       </div>`
                }
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Set up toolbar event listeners
document.getElementById('search-input').addEventListener('input', renderCheatSheet);
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderCheatSheet();
    });
});

// 4. Draft Logic
function populateDraftDropdowns() {
    const playerSelect = document.getElementById('draft-player-select');
    playerSelect.innerHTML = '<option value="" disabled selected>Choose player...</option>';
    
    // Only show undrafted players in dropdown
    const undrafted = allPlayers.filter(p => !draftedPlayers.has(p.player_name));
    undrafted.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.player_name;
        opt.textContent = `${p.player_name} (${p.position} - ${p.team_predict})`;
        playerSelect.appendChild(opt);
    });

    const teamSelect = document.getElementById('draft-team-select');
    teamSelect.innerHTML = '<option value="" disabled selected>Choose team...</option>';
    LEAGUE_TEAMS.forEach(team => {
        const opt = document.createElement('option');
        opt.value = team;
        opt.textContent = `${team} (${rosters[team].length} players)`;
        teamSelect.appendChild(opt);
    });
}

function setupDraftEvents() {
    document.getElementById('submit-pick-btn').addEventListener('click', () => {
        const playerSelect = document.getElementById('draft-player-select');
        const teamSelect = document.getElementById('draft-team-select');
        
        const playerName = playerSelect.value;
        const teamName = teamSelect.value;
        
        if (!playerName || !teamName) {
            alert("Please select both a player and a team.");
            return;
        }
        
        executeDraft(playerName, teamName);
    });

    document.getElementById('clear-draft-btn').addEventListener('click', () => {
        if (confirm("Are you sure you want to clear/reset the current draft?")) {
            draftedPlayers.clear();
            LEAGUE_TEAMS.forEach(t => rosters[t] = []);
            draftLog = [];
            
            renderCheatSheet();
            populateDraftDropdowns();
            renderRosters();
            renderDraftLog();
            
            // Reset standings view
            const standings = document.getElementById('standings-container');
            standings.innerHTML = `
                <div class="widget-placeholder">
                    <p>Draft players to teams to run the playoff simulation.</p>
                </div>
            `;
        }
    });
}

function executeDraft(playerName, teamName) {
    const playerObj = allPlayers.find(p => p.player_name === playerName);
    if (!playerObj) return;

    draftedPlayers.add(playerName);
    rosters[teamName].push(playerObj);
    draftLog.push({ player_name: playerName, team_name: teamName });

    renderCheatSheet();
    populateDraftDropdowns();
    renderRosters();
    renderDraftLog();
    
    // Automatically run playoff simulation in the background
    runPlayoffSimulation();
}

function draftPlayerQuick(playerName) {
    const teamSelectModal = document.getElementById('draft-team-select');
    // Prompt for team or automatically select the next team in the dropdown
    const teamName = prompt(`Draft ${playerName} to which league team?\nOptions:\n` + LEAGUE_TEAMS.join('\n'));
    if (!teamName) return;
    
    const matchedTeam = LEAGUE_TEAMS.find(t => t.toLowerCase() === teamName.trim().toLowerCase());
    if (!matchedTeam) {
        alert("Invalid team name.");
        return;
    }
    
    executeDraft(playerName, matchedTeam);
}

function undraftPlayer(playerName) {
    draftedPlayers.delete(playerName);
    
    // Remove from rosters
    LEAGUE_TEAMS.forEach(team => {
        rosters[team] = rosters[team].filter(p => p.player_name !== playerName);
    });
    
    // Remove from log
    draftLog = draftLog.filter(entry => entry.player_name !== playerName);
    
    renderCheatSheet();
    populateDraftDropdowns();
    renderRosters();
    renderDraftLog();
    
    if (draftedPlayers.size > 0) {
        runPlayoffSimulation();
    } else {
        const standings = document.getElementById('standings-container');
        standings.innerHTML = `
            <div class="widget-placeholder">
                <p>Draft players to teams to run the playoff simulation.</p>
            </div>
        `;
    }
}

function renderDraftLog() {
    const logList = document.getElementById('draft-log-list');
    logList.innerHTML = '';
    
    if (draftLog.length === 0) {
        logList.innerHTML = '<li class="log-placeholder">No picks recorded yet.</li>';
        return;
    }
    
    draftLog.forEach((pick, i) => {
        const li = document.createElement('li');
        li.innerHTML = `Pick ${i+1}: <strong>${pick.player_name}</strong> drafted to <em>${pick.team_name}</em>`;
        logList.appendChild(li);
    });
}

function renderRosters() {
    const container = document.getElementById('rosters-container');
    container.innerHTML = '';
    
    LEAGUE_TEAMS.forEach(team => {
        const card = document.createElement('div');
        card.className = 'roster-card';
        
        let playerRows = '';
        const teamPlayers = rosters[team];
        
        if (teamPlayers.length === 0) {
            playerRows = '<div class="roster-empty">No players drafted yet.</div>';
        } else {
            teamPlayers.forEach(p => {
                playerRows += `
                    <div class="roster-player-row">
                        <span class="roster-player-name">${p.player_name} (${p.position})</span>
                        <span class="roster-player-meta">${p.team_predict} - ${p.prev_fantasy_points.toFixed(1)} pts</span>
                    </div>
                `;
            });
        }
        
        card.innerHTML = `
            <h3>${team} (${teamPlayers.length})</h3>
            <div class="roster-players">
                ${playerRows}
            </div>
        `;
        container.appendChild(card);
    });
}

// 5. Playoff Simulation
function setupSimButton() {
    document.getElementById('sim-btn').addEventListener('click', () => {
        if (draftedPlayers.size === 0) {
            alert("Rosters are empty. Please draft some players first to run the season simulation.");
            return;
        }
        runPlayoffSimulation();
    });
}

async function runPlayoffSimulation() {
    const simIcon = document.getElementById('sim-icon');
    simIcon.classList.add('fa-spin');
    
    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rosters })
        });
        const res = await response.json();
        if (res.status === 'success') {
            renderStandings(res.leaderboard);
        } else {
            console.error(res.message);
        }
    } catch (err) {
        console.error(err);
    } finally {
        simIcon.classList.remove('fa-spin');
    }
}

function renderStandings(leaderboard) {
    const container = document.getElementById('standings-container');
    container.innerHTML = '';
    
    leaderboard.forEach((row, i) => {
        const div = document.createElement('div');
        div.className = 'standing-row';
        div.innerHTML = `
            <span class="standing-team">${i+1}. ${row.team}</span>
            <div class="standing-stats">
                <span>${row.avg_wins} W</span>
                <span class="standing-odds">${row.playoff_probability.toFixed(0)}% Odds</span>
            </div>
        `;
        container.appendChild(div);
    });
}

// 6. Google Drive Export
function setupExportEvents() {
    document.getElementById('export-btn').addEventListener('click', async () => {
        if (draftedPlayers.size === 0) {
            alert("Rosters are empty. Draft players before exporting.");
            return;
        }
        
        const btn = document.getElementById('export-btn');
        const folderInput = document.getElementById('drive-folder-id');
        const status = document.getElementById('export-status');
        
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Exporting...';
        status.className = 'status-msg hidden';
        
        try {
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rosters: rosters,
                    folder_id: folderInput.value.trim() || null
                })
            });
            const res = await response.json();
            if (res.status === 'success') {
                status.className = 'status-msg success';
                status.textContent = `Success! File ID: ${res.file_id}`;
            } else {
                status.className = 'status-msg error';
                status.textContent = `Export failed: ${res.message}`;
            }
        } catch (err) {
            console.error(err);
            status.className = 'status-msg error';
            status.textContent = 'Network error during Google Drive upload.';
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-brands fa-google-drive"></i> Export Rosters';
            status.classList.remove('hidden');
        }
    });
}

// Toggle custom dropdown
function toggleDropdown(event, playerName) {
    event.stopPropagation();
    const dropdownId = `dropdown-${playerName.replace(/[^a-zA-Z0-9]/g, '-')}`;
    const targetMenu = document.getElementById(dropdownId);
    
    // Close all other dropdowns
    document.querySelectorAll('.dropdown-menu').forEach(menu => {
        if (menu.id !== dropdownId) {
            menu.classList.add('hidden');
        }
    });
    
    // Toggle target
    if (targetMenu) {
        targetMenu.classList.toggle('hidden');
    }
}

// Close dropdowns when clicking anywhere else on page
document.addEventListener('click', () => {
    document.querySelectorAll('.dropdown-menu').forEach(menu => {
        menu.classList.add('hidden');
    });
});

