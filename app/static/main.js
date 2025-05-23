const inputField = document.getElementById("input-context");
const outputField = document.getElementById("output-context");

let intervalId = null;

inputField.addEventListener("input", () => {
    clearInterval(intervalId);
    const text = inputField.value.trim();

    intervalId = setInterval(() => {
        console.log("Input detected:", text);

        fetch("/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            outputField.innerText = data.translation;
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }, 500);
});

// <!DOCTYPE html>
// <html lang="en">
// <head>
//     <meta charset="UTF-8">
//     <meta name="viewport" content="width=device-width, initial-scale=1.0">
//     <title>Simple Translator</title>
//     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
//     <link href="{{ url_for('static', path='/styles.css') }}" rel="stylesheet">
// </head>
// <body>
//     <fieldset class="translator-box">
//         <legend>Simple Translator</legend>
//         <textarea name="" id="input-context" placeholder="Enter English text ..."></textarea>
//         <div class="result" id="output-container">
//             <p id="output-context">Translation will appear here...</p>
//             <i class="fa-regular fa-copy" onclick="copyToClipboard(this)"></i>
//         </div>
//     </fieldset>
//     <script src="{{ url_for('static', path='/main.js') }}"></script>
// </body>

// </html>
// Define a function to copy text to clipboard
function copyToClipboard(element) {
    const text = element.previousElementSibling.innerText;
    navigator.clipboard.writeText(text)
        .then(() => {
            console.log("Text copied to clipboard:", text);
        })
        .catch(err => {
            console.error("Error copying text:", err);
        });
}
