from IPython.display import Javascript, display

js = '''
document.querySelectorAll('[id*="shm-dropdown"]').forEach(el => el.remove());

const funcs = {
    'Spectral': [
        {name: 'PSD Welch', code: 'frequencies, psd = shmtools.psd_welch(x=data, fs=fs, nperseg=1024)'},
        {name: 'Spectrogram', code: 'f, t, Sxx = shmtools.spectrogram(x=data, fs=fs, nperseg=1024)'}
    ],
    'Time Series': [
        {name: 'AR Model', code: 'features, residuals = shmtools.ar_model(X=data, ar_order=15)'}
    ],
    'Outlier Detection': [
        {name: 'Learn PCA', code: 'model = shmtools.learn_pca(X=features, per_var=0.95)'},
        {name: 'Score PCA', code: 'scores, outliers = shmtools.score_pca(Y=test_data, model=model)'}
    ],
    'Filtering': [
        {name: 'Bandpass', code: 'filtered = shmtools.bandpass_filter(x=signal, lowcut=10, highcut=100, fs=fs)'}
    ]
};

const panel = document.createElement('div');
panel.id = 'shm-dropdown-panel';
panel.style.cssText = 'position:fixed; top:60px; right:20px; background:white; border:2px solid #007bff; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,0.15); z-index:1000; width:280px; font-family:Arial,sans-serif;';

const header = document.createElement('div');
header.innerHTML = '🔧 SHM Functions';
header.style.cssText = 'background:#007bff; color:white; padding:12px 16px; font-weight:bold; font-size:14px; border-radius:6px 6px 0 0; cursor:move;';

let isDragging = false, dragOffset = {x:0, y:0};
header.onmousedown = e => {
    isDragging = true;
    dragOffset.x = e.clientX - panel.offsetLeft;
    dragOffset.y = e.clientY - panel.offsetTop;
};
document.onmousemove = e => {
    if (isDragging) {
        panel.style.left = (e.clientX - dragOffset.x) + 'px';
        panel.style.top = (e.clientY - dragOffset.y) + 'px';
        panel.style.right = 'auto';
    }
};
document.onmouseup = () => isDragging = false;

panel.appendChild(header);

const content = document.createElement('div');
content.style.cssText = 'max-height:400px; overflow-y:auto;';

Object.keys(funcs).forEach(category => {
    const catHeader = document.createElement('div');
    catHeader.textContent = category;
    catHeader.style.cssText = 'padding:10px 16px; background:#f8f9fa; font-weight:bold; color:#495057; font-size:12px; border-bottom:1px solid #dee2e6;';
    content.appendChild(catHeader);
    
    funcs[category].forEach(func => {
        const item = document.createElement('div');
        item.textContent = func.name;
        item.style.cssText = 'padding:10px 16px; cursor:pointer; border-bottom:1px solid #f1f3f4; transition:background-color 0.2s;';
        
        item.onmouseover = () => item.style.backgroundColor = '#e3f2fd';
        item.onmouseout = () => item.style.backgroundColor = 'transparent';
        item.onclick = () => {
            navigator.clipboard.writeText(func.code);
            console.log('Copied to clipboard:', func.code);
        };
        
        content.appendChild(item);
    });
});

panel.appendChild(content);
document.body.appendChild(panel);
console.log('✅ SHM Extension loaded - click functions to copy code');
''';

display(Javascript(js))
print("🔧 SHM Function Selector loaded! Click functions to copy code to clipboard.")