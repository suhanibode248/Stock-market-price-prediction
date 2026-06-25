document.addEventListener("DOMContentLoaded", () => {
    const assetSelect = document.getElementById("asset-select");
    const modelSelect = document.getElementById("model-select");
    let chartInstance = null;

    async function loadData() {
        const symbol = assetSelect.value;
        const model = modelSelect.value;
        
        try {
            const response = await fetch(`/api/data/${symbol}?model_type=${model}`);
            const data = await response.json();
            
            if (data.error) {
                console.error(data.error);
                return;
            }
            
            updateKPIs(data);
            updateChart(data);
        } catch (err) {
            console.error("Failed to load data:", err);
        }
    }

    function updateKPIs(data) {
        document.getElementById("price-val").textContent = `$${data.latest_close.toFixed(2)}`;
        
        const priceDiff = data.latest_close - data.prev_close;
        const pricePct = (priceDiff / data.prev_close) * 100;
        const priceEl = document.getElementById("price-change");
        priceEl.textContent = `${priceDiff > 0 ? '+' : ''}${priceDiff.toFixed(2)} (${pricePct.toFixed(2)}%)`;
        priceEl.className = `card-subtext ${priceDiff >= 0 ? 'text-positive' : 'text-negative'}`;

        document.getElementById("rsi-val").textContent = data.latest_rsi.toFixed(2);
        
        if (data.metrics && data.metrics.da) {
            document.getElementById("da-val").textContent = `${data.metrics.da.toFixed(1)}%`;
        }
        
        if (data.next_pred) {
            document.getElementById("pred-val").textContent = `$${data.next_pred.toFixed(2)}`;
            const predDiff = data.next_pred - data.latest_close;
            const predEl = document.getElementById("pred-change");
            predEl.textContent = `Predicted Change: ${predDiff > 0 ? '+' : ''}${predDiff.toFixed(2)}`;
            predEl.className = `card-subtext ${predDiff >= 0 ? 'text-positive' : 'text-negative'}`;
        }
    }

    function updateChart(data) {
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        // Format dates
        const labels = data.dates.map(d => new Date(d).toLocaleDateString());
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Actual Close Price',
                        data: data.close,
                        borderColor: '#22c55e',
                        borderWidth: 2,
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: 'AI Prediction',
                        data: data.prediction,
                        borderColor: '#38bdf8',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.1,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: {
                        labels: { color: '#f8fafc' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    x: {
                        ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // Event Listeners
    assetSelect.addEventListener("change", loadData);
    modelSelect.addEventListener("change", loadData);

    // Initial Load
    loadData();
});
