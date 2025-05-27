async function submitMessage() {
  const message = document.getElementById("messageInput").value;
  const resultDiv = document.getElementById("result");

  if (message.trim() === "") {
    resultDiv.innerHTML = "<span style='color: #ffcccc;'>Please enter a message.</span>";
    return;
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    });

    if (!response.ok) {
      throw new Error("Server error: " + response.statusText);
    }

    const data = await response.json();
    resultDiv.innerHTML = "Prediction: <strong>" + data.prediction + "</strong>";
  } catch (error) {
    resultDiv.innerHTML = "<span style='color: #ffcccc;'>Error: " + error.message + "</span>";
  }
}
