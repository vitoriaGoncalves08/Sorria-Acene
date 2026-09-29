// A webcam precisa ficar aberta continuamente, e popups da extensão fecham
// quando perdem o foco (o que mataria o stream de vídeo). Por isso o botão
// da extensão abre uma aba dedicada em vez de um popup.
chrome.action.onClicked.addListener(() => {
  chrome.tabs.create({ url: chrome.runtime.getURL("src/app/app.html") });
});
