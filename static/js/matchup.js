document.addEventListener('DOMContentLoaded', () => {
    let currentMatchupIndex = 0;
    let matchupData = null;
    let activeSubSource = null; // { teamKey: 'team1'|'team2', isStarter: true|false, index: number, slot: string, player: object }

    // DOM Elements
    const pillsBar = document.getElementById('matchup-pills-bar');
    const t1Name = document.getElementById('t1-name');
    const t2Name = document.getElementById('t2-name');
    const t1NnPts = document.getElementById('t1-nn-pts');
    const t1YahooPts = document.getElementById('t1-yahoo-pts');
    const t2NnPts = document.getElementById('t2-nn-pts');
    const t2YahooPts = document.getElementById('t2-yahoo-pts');
    const consensusBadge = document.getElementById('consensus-badge');
    const consensusText = document.getElementById('consensus-text');

    const nnProbT1Val = document.getElementById('nn-prob-t1-val');
    const nnProbT2Val = document.getElementById('nn-prob-t2-val');
    const nnProgressBar = document.getElementById('nn-progress-bar');

    const yahooProbT1Val = document.getElementById('yahoo-prob-t1-val');
    const yahooProbT2Val = document.getElementById('yahoo-prob-t2-val');
    const yahooProgressBar = document.getElementById('yahoo-progress-bar');

    const t1ColName = document.getElementById('t1-col-name');
    const t2ColName = document.getElementById('t2-col-name');
    const t1StartersTbody = document.getElementById('t1-starters-tbody');
    const t1BenchTbody = document.getElementById('t1-bench-tbody');
    const t2StartersTbody = document.getElementById('t2-starters-tbody');
    const t2BenchTbody = document.getElementById('t2-bench-tbody');

    const subAlertBar = document.getElementById('sub-active-alert');
    const subSourcePlayer = document.getElementById('sub-source-player');
    const subSourceSlot = document.getElementById('sub-source-slot');
    const cancelSubBtn = document.getElementById('cancel-sub-btn');

    const autoOptimizeBtn = document.getElementById('auto-optimize-btn');
    const resetLineupBtn = document.getElementById('reset-lineup-btn');
    const intelGrid = document.getElementById('intel-cards-grid');

    // Fetch and Load Matchup
    async function loadMatchup(index = 0) {
        currentMatchupIndex = index;
        clearSubState();
        try {
            const resp = await fetch(`/api/matchup_data?index=${index}`);
            const json = await resp.json();
            if (json.status === 'success') {
                matchupData = json.data;
                renderMatchupPills();
                renderScoreboard();
                renderRosters();
                renderIntelCards();
            } else {
                console.error("Failed to load matchup:", json.message);
            }
        } catch (e) {
            console.error("Error loading matchup data:", e);
        }
    }

    // Render Matchup Selector Pills
    function renderMatchupPills() {
        if (!matchupData || !matchupData.matchups_list) return;
        pillsBar.innerHTML = '';
        matchupData.matchups_list.forEach((m) => {
            const pill = document.createElement('div');
            pill.className = `matchup-pill ${m.index === currentMatchupIndex ? 'active' : ''} ${m.is_my_matchup ? 'my-matchup' : ''}`;
            pill.innerHTML = `
                ${m.is_my_matchup ? '<i class="fa-solid fa-star text-warning"></i>' : ''}
                <span>${m.team1_name.split(' ')[0]} vs ${m.team2_name.split(' ')[0]}</span>
            `;
            pill.addEventListener('click', () => loadMatchup(m.index));
            pillsBar.appendChild(pill);
        });
    }

    // Render Scoreboard & Win Probabilities
    function renderScoreboard() {
        if (!matchupData) return;
        const t1 = matchupData.team1;
        const t2 = matchupData.team2;
        const sum = matchupData.summary;

        t1Name.textContent = t1.name;
        t2Name.textContent = t2.name;
        t1ColName.textContent = t1.name;
        t2ColName.textContent = t2.name;

        document.getElementById('t1-user-badge').style.display = t1.is_my_team ? 'inline-block' : 'none';
        document.getElementById('t2-user-badge').style.display = t2.is_my_team ? 'inline-block' : 'none';

        t1NnPts.textContent = sum.team1.nn_pts.toFixed(2);
        t1YahooPts.textContent = sum.team1.yahoo_pts.toFixed(2);
        t2NnPts.textContent = sum.team2.nn_pts.toFixed(2);
        t2YahooPts.textContent = sum.team2.yahoo_pts.toFixed(2);

        // Win Probabilities
        nnProbT1Val.textContent = `${sum.team1.nn_win_prob}%`;
        nnProbT2Val.textContent = `${sum.team2.nn_win_prob}%`;
        nnProgressBar.style.width = `${sum.team1.nn_win_prob}%`;

        yahooProbT1Val.textContent = `${sum.team1.yahoo_win_prob}%`;
        yahooProbT2Val.textContent = `${sum.team2.yahoo_win_prob}%`;
        yahooProgressBar.style.width = `${sum.team1.yahoo_win_prob}%`;

        // Consensus / Disagreement Badge
        consensusText.textContent = sum.divergence_summary;
        if (!sum.models_agree) {
            consensusBadge.classList.add('disagree');
            consensusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${sum.divergence_summary}</span>`;
        } else {
            consensusBadge.classList.remove('disagree');
            consensusBadge.innerHTML = `<i class="fa-solid fa-scale-balanced"></i> <span>${sum.divergence_summary}</span>`;
        }
    }

    // Get Slot Class
    function getSlotClass(slot) {
        switch (slot.toUpperCase()) {
            case 'QB': return 'slot-qb';
            case 'RB': return 'slot-rb';
            case 'WR': return 'slot-wr';
            case 'TE': return 'slot-te';
            case 'W/R/T':
            case 'FLEX': return 'slot-flex';
            case 'DEF': return 'slot-def';
            default: return 'slot-bn';
        }
    }

    // Render Starter / Bench Tables
    function renderRosters() {
        if (!matchupData) return;
        renderTeamRoster('team1', t1StartersTbody, t1BenchTbody);
        renderTeamRoster('team2', t2StartersTbody, t2BenchTbody);
    }

    function renderTeamRoster(teamKey, startersTbody, benchTbody) {
        const teamObj = matchupData[teamKey];
        startersTbody.innerHTML = '';
        benchTbody.innerHTML = '';

        // Starters
        teamObj.starters.forEach((s, idx) => {
            const p = s.player;
            const tr = document.createElement('tr');
            tr.dataset.team = teamKey;
            tr.dataset.type = 'starter';
            tr.dataset.index = idx;
            tr.dataset.slot = s.slot;

            if (activeSubSource && activeSubSource.teamKey === teamKey && activeSubSource.isStarter && activeSubSource.index === idx) {
                tr.classList.add('sub-source-row');
            }

            tr.innerHTML = `
                <td><span class="slot-badge ${getSlotClass(s.slot)}">${s.slot}</span></td>
                <td>
                    <div class="player-cell-main">
                        <span class="player-title">${p.player_name}</span>
                        <span class="player-subtitle">${p.pos} • ${p.team}</span>
                    </div>
                </td>
                <td>
                    <span class="player-pts nn-pts-val">${p.nn_proj.toFixed(2)}</span>
                    <span class="proj-range">±${p.nn_std.toFixed(1)}</span>
                </td>
                <td>
                    <span class="player-pts yahoo-pts-val">${p.yahoo_proj.toFixed(2)}</span>
                    <span class="proj-range">${p.yahoo_min.toFixed(1)} - ${p.yahoo_max.toFixed(1)}</span>
                </td>
                <td>
                    <span class="div-badge div-${p.divergence_type}">${p.divergence_label}</span>
                </td>
                <td>
                    <button class="btn-sub starter-sub-btn" data-team="${teamKey}" data-index="${idx}" data-slot="${s.slot}">
                        <i class="fa-solid fa-arrow-right-arrow-left"></i> SUB
                    </button>
                </td>
            `;
            startersTbody.appendChild(tr);
        });

        // Bench
        teamObj.bench.forEach((b, idx) => {
            const p = b.player;
            const tr = document.createElement('tr');
            tr.dataset.team = teamKey;
            tr.dataset.type = 'bench';
            tr.dataset.index = idx;
            tr.dataset.pos = p.pos;

            // Highlight if active sub source is compatible
            if (activeSubSource && activeSubSource.teamKey === teamKey && activeSubSource.isStarter) {
                const targetSlot = activeSubSource.slot;
                const canFit = isEligibleForSlot(p.pos, targetSlot);
                if (canFit) {
                    tr.classList.add('sub-target-eligible');
                }
            }

            tr.innerHTML = `
                <td><span class="slot-badge ${getSlotClass(p.pos)}">${p.pos}</span></td>
                <td>
                    <div class="player-cell-main">
                        <span class="player-title">${p.player_name}</span>
                        <span class="player-subtitle">${p.team}</span>
                    </div>
                </td>
                <td>
                    <span class="player-pts nn-pts-val">${p.nn_proj.toFixed(2)}</span>
                    <span class="proj-range">±${p.nn_std.toFixed(1)}</span>
                </td>
                <td>
                    <span class="player-pts yahoo-pts-val">${p.yahoo_proj.toFixed(2)}</span>
                    <span class="proj-range">${p.yahoo_min.toFixed(1)} - ${p.yahoo_max.toFixed(1)}</span>
                </td>
                <td>
                    <span class="div-badge div-${p.divergence_type}">${p.divergence_label}</span>
                </td>
                <td>
                    <button class="btn-sub bench-sub-btn" data-team="${teamKey}" data-index="${idx}">
                        <i class="fa-solid fa-arrow-up"></i> SUB IN
                    </button>
                </td>
            `;
            benchTbody.appendChild(tr);
        });

        attachSubEvents(teamKey);
    }

    function isEligibleForSlot(playerPos, slot) {
        slot = slot.toUpperCase();
        playerPos = playerPos.toUpperCase();
        if (slot === playerPos) return true;
        if ((slot === 'W/R/T' || slot === 'FLEX') && ['RB', 'WR', 'TE'].includes(playerPos)) return true;
        return false;
    }

    // Attach Click Events for Sub Swaps
    function attachSubEvents(teamKey) {
        const teamObj = matchupData[teamKey];

        // Starter Sub Button click
        document.querySelectorAll(`.starter-sub-btn[data-team="${teamKey}"]`).forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(btn.dataset.index);
                const slot = btn.dataset.slot;
                const player = teamObj.starters[idx].player;

                activeSubSource = {
                    teamKey: teamKey,
                    isStarter: true,
                    index: idx,
                    slot: slot,
                    player: player
                };

                subSourcePlayer.textContent = player.player_name;
                subSourceSlot.textContent = slot;
                subAlertBar.style.display = 'flex';

                renderRosters();
            });
        });

        // Bench Sub Button click or clicking eligible bench row
        document.querySelectorAll(`.bench-sub-btn[data-team="${teamKey}"]`).forEach(btn => {
            btn.addEventListener('click', (e) => {
                const benchIdx = parseInt(btn.dataset.index);
                handleBenchSelection(teamKey, benchIdx);
            });
        });

        document.querySelectorAll(`tr.sub-target-eligible[data-team="${teamKey}"]`).forEach(row => {
            row.addEventListener('click', (e) => {
                if (e.target.closest('button')) return;
                const benchIdx = parseInt(row.dataset.index);
                handleBenchSelection(teamKey, benchIdx);
            });
        });
    }

    // Execute Swap
    async function handleBenchSelection(teamKey, benchIdx) {
        const teamObj = matchupData[teamKey];
        const benchPlayer = teamObj.bench[benchIdx].player;

        if (activeSubSource && activeSubSource.teamKey === teamKey && activeSubSource.isStarter) {
            const starterIdx = activeSubSource.index;
            const starterSlot = activeSubSource.slot;
            const starterPlayer = teamObj.starters[starterIdx].player;

            if (!isEligibleForSlot(benchPlayer.pos, starterSlot)) {
                alert(`${benchPlayer.player_name} (${benchPlayer.pos}) is not eligible for the ${starterSlot} slot.`);
                return;
            }

            // Perform Swap
            teamObj.starters[starterIdx].player = benchPlayer;
            teamObj.bench[benchIdx].player = starterPlayer;

            clearSubState();
            await recalculateMatchup();
        } else {
            // If user clicked bench without picking starter: find first eligible slot
            const eligibleStarterIdx = teamObj.starters.findIndex(s => isEligibleForSlot(benchPlayer.pos, s.slot));
            if (eligibleStarterIdx !== -1) {
                const starterPlayer = teamObj.starters[eligibleStarterIdx].player;
                teamObj.starters[eligibleStarterIdx].player = benchPlayer;
                teamObj.bench[benchIdx].player = starterPlayer;
                await recalculateMatchup();
            } else {
                alert(`No open or compatible starting slot for ${benchPlayer.player_name} (${benchPlayer.pos}).`);
            }
        }
    }

    function clearSubState() {
        activeSubSource = null;
        subAlertBar.style.display = 'none';
        renderRosters();
    }

    cancelSubBtn.addEventListener('click', clearSubState);

    // Recalculate Projections and Probabilities
    async function recalculateMatchup() {
        try {
            const resp = await fetch('/api/matchup/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    team1_starters: matchupData.team1.starters,
                    team2_starters: matchupData.team2.starters
                })
            });
            const json = await resp.json();
            if (json.status === 'success') {
                matchupData.summary = json.summary;
                renderScoreboard();
                renderRosters();
                renderIntelCards();
            }
        } catch (e) {
            console.error("Error recalculating projections:", e);
        }
    }

    // Auto-Optimize Lineup with NN
    autoOptimizeBtn.addEventListener('click', async () => {
        if (!matchupData) return;
        ['team1', 'team2'].forEach(tKey => {
            const teamObj = matchupData[tKey];
            const allPlayers = [...teamObj.starters.map(s => s.player), ...teamObj.bench.map(b => b.player)];

            // Group by pos
            const byPos = { 'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DEF': [], 'OTHER': [] };
            allPlayers.forEach(p => {
                if (byPos[p.pos]) byPos[p.pos].push(p);
                else byPos['OTHER'].push(p);
            });

            for (let k in byPos) byPos[k].sort((a, b) => b.nn_proj - a.nn_proj);

            const newStarters = [];
            if (byPos['QB'].length) newStarters.push({ slot: 'QB', player: byPos['QB'].shift() });
            for (let i = 0; i < 2; i++) if (byPos['RB'].length) newStarters.push({ slot: 'RB', player: byPos['RB'].shift() });
            for (let i = 0; i < 2; i++) if (byPos['WR'].length) newStarters.push({ slot: 'WR', player: byPos['WR'].shift() });
            if (byPos['TE'].length) newStarters.push({ slot: 'TE', player: byPos['TE'].shift() });

            const flexPool = [...byPos['RB'], ...byPos['WR'], ...byPos['TE']].sort((a, b) => b.nn_proj - a.nn_proj);
            for (let i = 0; i < 2; i++) {
                if (flexPool.length) {
                    const fp = flexPool.shift();
                    const posIdx = byPos[fp.pos].indexOf(fp);
                    if (posIdx > -1) byPos[fp.pos].splice(posIdx, 1);
                    newStarters.push({ slot: 'W/R/T', player: fp });
                }
            }

            if (byPos['DEF'].length) newStarters.push({ slot: 'DEF', player: byPos['DEF'].shift() });

            const newBench = [];
            for (let k in byPos) byPos[k].forEach(p => newBench.push({ slot: 'BN', player: p }));

            teamObj.starters = newStarters;
            teamObj.bench = newBench;
        });

        await recalculateMatchup();
    });

    // Reset Lineup Button
    resetLineupBtn.addEventListener('click', () => loadMatchup(currentMatchupIndex));

    // Render Intelligence & Disagreement Cards
    function renderIntelCards() {
        if (!matchupData) return;
        intelGrid.innerHTML = '';
        const allStarters = [...matchupData.team1.starters, ...matchupData.team2.starters];

        // Filter players with significant divergence
        const diverged = allStarters.filter(s => Math.abs(s.player.diff) >= 1.5)
            .sort((a, b) => Math.abs(b.player.diff) - Math.abs(a.player.diff));

        if (diverged.length === 0) {
            intelGrid.innerHTML = `<p class="text-secondary" style="grid-column: 1/-1;">Both models are largely in consensus across all starters for this matchup (divergences < 1.5 pts).</p>`;
            return;
        }

        diverged.forEach(s => {
            const p = s.player;
            const card = document.createElement('div');
            card.className = 'intel-card';
            card.innerHTML = `
                <div class="intel-card-header">
                    <span class="intel-player-name">${p.player_name} (${p.pos} - ${p.team})</span>
                    <span class="div-badge div-${p.divergence_type}">${p.divergence_label}</span>
                </div>
                <div class="intel-comparison">
                    <span><strong>NN:</strong> ${p.nn_proj.toFixed(2)} pts</span>
                    <span><strong>Yahoo:</strong> ${p.yahoo_proj.toFixed(2)} pts</span>
                    <span><strong>Delta:</strong> ${p.diff > 0 ? '+' : ''}${p.diff.toFixed(2)} pts</span>
                </div>
                <div class="intel-reasons">
                    ${(p.reasons || []).map(r => `<span class="intel-tag"><i class="fa-solid fa-chart-simple"></i> ${r}</span>`).join('')}
                </div>
            `;
            intelGrid.appendChild(card);
        });
    }

    // Initial Load
    loadMatchup(0);
});
