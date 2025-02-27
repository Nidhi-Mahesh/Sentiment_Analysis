function analyzeSentiment() {
    const text = document.getElementById('text-input').value;
    
    if (!text.trim()) {
        alert('Please enter some text to analyze');
        return;
    }

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = data.sentiment;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing the text');
    });
} 