// Global State
let allPlayers = [];
let draftedPlayers = new Set(); // Set of player_name
let rosters = {}; // Map of team_name -> [player_dicts]
let draftLog = []; // Array of {player_name, team_name}
let completedDraftData = null;

let LEAGUE_TEAMS = [];
let MY_TEAM = '';

// Initialize Rosters
function initializeRosters() {
    rosters = {};
    LEAGUE_TEAMS.forEach(team => {
        rosters[team] = [];
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    initializeTabs();
    initializeOlineSliders();
    await fetchLeagueSettings();
    initializeRosters();
    await fetchPlayers();
    await fetchCompletedDraft();
    setupDraftEvents();
    setupExportEvents();
    setupSimButton();
    setupDraftFilterEvents();
});

async function fetchLeagueSettings() {
    try {
        const response = await fetch('/api/league_settings');
        const res = await response.json();
        if (res.status === 'success') {
            LEAGUE_TEAMS = res.settings.league_teams;
            MY_TEAM = res.settings.my_team;
        }
    } catch (e) {
        console.error("Failed to load settings", e);
    }
}

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
    
    // Automatically run playoff simulation and strategy in the background
    runPlayoffSimulation();
    fetchStrategyAdvice();
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
    fetchStrategyAdvice();
}

async function fetchCompletedDraft() {
    try {
        const response = await fetch('/api/completed_draft');
        const res = await response.json();
        if (res.status === 'success' && res.data) {
            completedDraftData = res.data;
            
            // Populate drafted players and log
            draftLog = completedDraftData.draft_log;
            draftedPlayers.clear();
            draftLog.forEach(pick => draftedPlayers.add(pick.player_name));
            
            // Fetch rosters
            const rResp = await fetch('/api/rosters');
            const rJson = await rResp.json();
            if (rJson.status === 'success' && rJson.rosters) {
                // Enrich roster players with stats if available
                LEAGUE_TEAMS.forEach(team => {
                    rosters[team] = (rJson.rosters[team] || []).map(rawP => {
                        const matched = allPlayers.find(ap => ap.player_name.toLowerCase() === rawP.player_name.toLowerCase());
                        return matched || {
                            player_name: rawP.player_name,
                            position: rawP.position,
                            team_predict: rawP.nfl_team || 'UNK',
                            prev_fantasy_points: 120.0,
                            predicted_pts: 120.0
                        };
                    });
                });
            }

            renderDraftGrades(completedDraftData.team_grades);
            populateTeamFilterDropdown();
            renderDraftResultsTable(completedDraftData.draft_log);
            renderRosters();
            renderCheatSheet();
            runPlayoffSimulation();
        }
    } catch (e) {
        console.error("Failed to load completed draft data", e);
    }
}

function renderDraftGrades(grades) {
    const grid = document.getElementById('draft-grades-grid');
    if (!grid || !grades) return;
    grid.innerHTML = '';

    grades.forEach(t => {
        const card = document.createElement('div');
        card.className = `grade-card ${t.is_my_team ? 'my-team-card' : ''}`;
        
        let gradeClass = 'grade-b';
        if (t.grade.startsWith('A')) gradeClass = 'grade-a';
        else if (t.grade.startsWith('C')) gradeClass = 'grade-c';

        card.innerHTML = `
            <div class="grade-card-header">
                <span class="grade-team-name">${t.rank}. ${t.team_name}</span>
                <span class="grade-badge-pill ${gradeClass}">${t.grade}</span>
            </div>
            <div class="grade-meta-row">
                <span><strong>VORP:</strong> ${t.total_vorp > 0 ? '+'+t.total_vorp : t.total_vorp}</span>
                <span><strong>Proj Pts:</strong> ${t.total_pts}</span>
            </div>
            <div class="grade-steals-box">
                <span class="steal-text"><i class="fa-solid fa-gem"></i> Steal: ${t.best_steal ? t.best_steal.player_name + ' (Pick ' + t.best_steal.overall_pick + ')' : 'N/A'}</span>
                <span class="reach-text"><i class="fa-solid fa-arrow-trend-down"></i> Reach: ${t.biggest_reach ? t.biggest_reach.player_name + ' (Pick ' + t.biggest_reach.overall_pick + ')' : 'N/A'}</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

function populateTeamFilterDropdown() {
    const filterSelect = document.getElementById('draft-filter-team');
    if (!filterSelect) return;
    filterSelect.innerHTML = '<option value="ALL">All 12 Teams</option>';
    LEAGUE_TEAMS.forEach(team => {
        const opt = document.createElement('option');
        opt.value = team;
        opt.textContent = team;
        filterSelect.appendChild(opt);
    });
}

function renderDraftResultsTable(log) {
    const tbody = document.getElementById('draft-results-tbody');
    if (!tbody || !log) return;
    tbody.innerHTML = '';

    const searchTerm = (document.getElementById('draft-search-input')?.value || '').toLowerCase().trim();
    const teamFilter = document.getElementById('draft-filter-team')?.value || 'ALL';

    const filtered = log.filter(p => {
        const matchSearch = !searchTerm || p.player_name.toLowerCase().includes(searchTerm) || p.fantasy_team.toLowerCase().includes(searchTerm) || p.position.toLowerCase().includes(searchTerm);
        const matchTeam = (teamFilter === 'ALL') || (p.fantasy_team === teamFilter);
        return matchSearch && matchTeam;
    });

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center" style="padding: 24px; color: var(--text-secondary);">No draft picks matching your filter.</td></tr>`;
        return;
    }

    filtered.forEach(p => {
        const tr = document.createElement('tr');
        if (p.is_my_pick) tr.classList.add('my-pick-row');

        tr.innerHTML = `
            <td><strong class="font-mono">#${p.overall_pick}</strong></td>
            <td><span class="text-secondary">Round ${p.round} (${p.pick_in_round})</span></td>
            <td>
                <strong>${p.player_name}</strong>
                <span class="badge pos-${p.position}" style="margin-left: 6px;">${p.position}</span>
                <span class="text-secondary" style="font-size: 11px; margin-left: 4px;">${p.nfl_team}</span>
            </td>
            <td>
                <span class="${p.is_my_pick ? 'text-success font-bold' : ''}">
                    ${p.is_my_pick ? '<i class="fa-solid fa-star text-warning"></i> ' : ''}${p.fantasy_team}
                </span>
            </td>
            <td><span class="font-mono font-bold text-accent">${p.nn_proj_pts.toFixed(1)}</span></td>
            <td><span class="font-mono">${p.nn_vorp > 0 ? '+'+p.nn_vorp.toFixed(1) : p.nn_vorp.toFixed(1)}</span></td>
            <td>
                <span style="font-size: 12px;">
                    <strong class="text-accent">NN:</strong> #${p.nn_adp.toFixed(0)} 
                    <span class="text-secondary">|</span> 
                    <strong style="color: #c084fc;">Yahoo:</strong> #${p.yahoo_adp.toFixed(0)}
                </span>
            </td>
            <td>
                <span class="val-badge val-${p.val_type}">${p.val_badge}</span>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function setupDraftFilterEvents() {
    const searchInput = document.getElementById('draft-search-input');
    const teamSelect = document.getElementById('draft-filter-team');

    if (searchInput) {
        searchInput.addEventListener('input', () => {
            if (completedDraftData) renderDraftResultsTable(completedDraftData.draft_log);
        });
    }

    if (teamSelect) {
        teamSelect.addEventListener('change', () => {
            if (completedDraftData) renderDraftResultsTable(completedDraftData.draft_log);
        });
    }
}

function renderDraftLog() {
    if (completedDraftData) {
        renderDraftResultsTable(completedDraftData.draft_log);
    }
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

// Strategy Engine Polling
async function fetchStrategyAdvice() {
    try {
        const response = await fetch('/api/strategy_advice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rosters: rosters,
                draft_log: draftLog,
                all_players: allPlayers
            })
        });
        const res = await response.json();
        if (res.status === 'success') {
            renderStrategyAdvice(res.strategy);
        }
    } catch (e) {
        console.error("Failed to fetch strategy advice", e);
    }
}

function renderStrategyAdvice(strategy) {
    document.getElementById('strat-current-pick').textContent = strategy.current_pick;
    document.getElementById('strat-current-round').textContent = strategy.current_round;
    document.getElementById('strat-on-clock').textContent = strategy.team_on_clock;
    
    const untilTurnStr = strategy.is_my_turn ? "ON THE CLOCK!" : 
                        (strategy.picks_until_my_turn === -1 ? "N/A" : strategy.picks_until_my_turn);
    const untilSpan = document.getElementById('strat-picks-until');
    untilSpan.textContent = untilTurnStr;
    if (strategy.is_my_turn) {
        untilSpan.style.color = '#69BE28';
        untilSpan.style.fontWeight = 'bold';
    } else {
        untilSpan.style.color = 'inherit';
        untilSpan.style.fontWeight = 'inherit';
    }
    
    const needsCounts = strategy.roster_needs.counts || {};
    const needsList = strategy.roster_needs.needs || [];
    let needsHtml = `
        <div style="display:flex; gap: 10px; margin-bottom:10px;">
            <span class="badge pos-QB">QB: ${needsCounts.QB||0}</span>
            <span class="badge pos-RB">RB: ${needsCounts.RB||0}</span>
            <span class="badge pos-WR">WR: ${needsCounts.WR||0}</span>
            <span class="badge pos-TE">TE: ${needsCounts.TE||0}</span>
        </div>
        <p><strong>Primary Needs:</strong> ${needsList.join(', ') || 'None (Best Player Available)'}</p>
    `;
    document.getElementById('strat-needs-text').innerHTML = needsHtml;
    
    const warningsPanel = document.getElementById('strategy-warnings-panel');
    if (strategy.tier_warnings && strategy.tier_warnings.length > 0) {
        warningsPanel.innerHTML = strategy.tier_warnings.map(w => `
            <div class="strategy-warning-card">
                <i class="fa-solid fa-triangle-exclamation"></i> ${w}
            </div>
        `).join('');
    } else {
        warningsPanel.innerHTML = '';
    }
    
    const recsPanel = document.getElementById('strategy-recs-panel');
    if (strategy.recommendations && strategy.recommendations.length > 0) {
        recsPanel.innerHTML = strategy.recommendations.map(r => {
            const p = r.player;
            const escapedName = p.player_name.replace(/'/g, "\\'");
            return `
                <div class="strategy-rec-card">
                    <div class="rec-header">
                        <h4>${p.player_name} <span class="badge pos-${p.position}">${p.position}</span></h4>
                        <span class="rec-reason-badge">${r.reason_title}</span>
                    </div>
                    <p class="rec-desc">${r.reason}</p>
                    <div class="rec-meta">
                        <span><strong>Pts:</strong> ${p.predicted_pts.toFixed(1)}</span>
                        <span><strong>VORP:</strong> ${p.vorp > 0 ? '+'+p.vorp.toFixed(1) : p.vorp.toFixed(1)}</span>
                        <span><strong>ADP:</strong> ${p.predicted_adp.toFixed(1)}</span>
                    </div>
                    <button class="btn btn-primary btn-small" onclick="executeDraft('${escapedName}', '${MY_TEAM}')" style="margin-top:10px; width:100%;">
                        <i class="fa-solid fa-plus"></i> Draft to My Team
                    </button>
                </div>
            `;
        }).join('');
    } else {
        recsPanel.innerHTML = '<div class="roster-empty">No recommendations available.</div>';
    }
}

