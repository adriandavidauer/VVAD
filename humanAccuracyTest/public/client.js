console.log('Client-side code running');

const id = document.getElementById('user_id').innerText;
const video = document.getElementById('videoPlayer');
const replayButton = document.getElementById('replay');
const notSpeakingButton = document.getElementById('not_speaking');
const speakingButton = document.getElementById('speaking');
const text = document.getElementById('text');
const your_statistics = document.getElementById('your_statistics');
const all_statistic = document.getElementById('all_statistics');



speakingButton.addEventListener('click', function(e) {
  console.log('button was clicked for' + id);
  sendClick(true)
});

notSpeakingButton.addEventListener('click', function(e) {
  console.log('button was clicked for' + id);
  sendClick(false)
});

replayButton.addEventListener('click', function(e){
  console.log("replay video")
  video.play()
});

function requestNewVideo(){

  video.load()
  video.play()

  // fetch('/video/'+id, {method: 'GET', headers: {'Content-Type': 'video/mp4'}})
  // .then(function(response) {
  //   if(response.ok) {
  //     //reload videoplayer
  //     var video = document.getElementById('videoPlayer');
  //     video.load()
  //     video.play()
  //   }
  // });
}
function showEndPage(){
  speakingButton.style.visibility="hidden"
  notSpeakingButton.style.visibility="hidden"
  video.style.visibility = "hidden"
  replayButton.style.visibility = "hidden"

  text.style.visibility = "visible"
  // your_statistics.style.visibility = "visible"
  // all_statistic.style.visibility = "visible"
}

function sendClick(c){
    var data = {user:id, class:c}
    fetch('/clicked/', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)})
    .then(function(response) {
      if(response.status == 200) {
        console.log('Click was recorded');
        requestNewVideo()
        return;
      }else if(response.status == 205){
        console.log("Reached end")
        showEndPage()
      }
      throw new Error('Request failed.');
    })
    .catch(function(error) {
      console.log(error);
    });
}