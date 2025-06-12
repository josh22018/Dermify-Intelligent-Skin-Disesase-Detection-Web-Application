async function askQuestion(disease) {
  const query = document.getElementById("query").value;
  const responseBox = document.getElementById("response");
  const loader = document.getElementById("chat-loader");

  loader.style.display = "flex";
  responseBox.innerText = "";

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ disease: disease, query: query })
  });

  const data = await response.json();
  loader.style.display = "none";
  responseBox.innerText = data.response;
}
