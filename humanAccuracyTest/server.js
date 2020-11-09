const express = require('express')
const fs = require('fs')
const path = require('path')
const app = express()
const MongoClient = require('mongodb').MongoClient;
const uniqid = require('uniqid');
var glob = require("glob")
var _ = require("lodash")


app.use(express.static(path.join(__dirname, 'public')))
app.use(express.static(path.join(__dirname, 'assets')))

// Parse URL-encoded bodies (as sent by HTML forms)
app.use(express.urlencoded());

// Parse JSON bodies (as sent by API clients)
app.use(express.json());

// connect to the db and start the express server
var client;
var db;
var classifications;
// ***Replace the URL below with the URL for your database***
//const url =  'mongodb://user:password@mongo_address:mongo_port/databaseName';
// E.g. for option 2) above this will be:
const url =  'mongodb://localhost:27017/';

var indexTemplate;
fs.readFile(path.join(__dirname + '/index.htm'), "utf8", function(err, data) {
  if (err){throw err} 
  indexTemplate = data
});

var allVids = fs.readdirSync(path.join(__dirname + '/assets/videoSamples'))
glob(path.join(__dirname + '/assets/videoSamples/**/*.mp4'), 'nonull', function (er, files) { 
  if (er){throw err}
  allVids = files
})
var files = {}
var activeFile = {}

app.get('/', function(req, res) {
  index = indexTemplate
  //set an user_id(put it into the index to send it back on button click)
  id = uniqid()
  index = index.replace(/USER\_ID/g, id.toString())
  // construct list of samples(shuffled) - for each user
  files[id] = _.shuffle(allVids)

  res.send(index)
  // res.sendFile(path.join(__dirname + '/index.htm'))
})

app.get('/video/:id', function(req, res) {
  var id = req.params.id;
  const path = files[id].pop()
  activeFile[id] = path

  console.log("VIDEO REQUEST")
  console.log("Currently showing: " + activeFile[id])
  const stat = fs.statSync(path)
  const fileSize = stat.size
  // const range = req.headers.range
  // const range = false //mayberemove
  

  // if (range) {
  //   const parts = range.replace(/bytes=/, "").split("-")
  //   const start = parseInt(parts[0], 10)
  //   const end = parts[1]
  //     ? parseInt(parts[1], 10)
  //     : fileSize-1

  //   const chunksize = (end-start)+1
  //   const file = fs.createReadStream(path, {start, end})
  //   const head = {
  //     'Content-Range': `bytes ${start}-${end}/${fileSize}`,
  //     'Accept-Ranges': 'bytes',
  //     'Content-Length': chunksize,
  //     'Content-Type': 'video/mp4',
  //   }

  //   res.writeHead(206, head)
  //   file.pipe(res)
  // } else {
    const head = {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
    }
    res.writeHead(200, head)
    fs.createReadStream(path).pipe(res)
  // }
})

function onConnect(err, client){
  if(err) {
  //  console.log(err);
   //console.log('trying again:' + err.toString())
   MongoClient.connect(url, onConnect );
  }else{
    console.log('Connected sucessufully to database')
    db = client.db("humanAccuracy");
    classifications = db.collection("classifications")
    
    // start the express web server listening on 3000
    app.listen(3000, function () {
      console.log('Listening on port 3000!')
      
    });
  }
}
app.post('/clicked', (req, res) => {
  // save settings and show next until there is no next in list.
  var user = req.body.user
  var c = req.body.class

  console.log("User: " + user)
  console.log("Classification: " + c)
  var gt;
  console.log("Path: " + activeFile[user])
  if(activeFile[user].includes("positive")){
    gt = true
  }else{
    gt = false
  }
  var splitted = activeFile[user].split('/')
  var numSample = splitted[splitted.length -1].split('.')[0]
  console.log("SAMPLENUM: " + numSample)
  console.log("GT: " + gt)
  console.log("-------------")
  var classification = {user_id:user, sample_num:numSample, ground_truth:gt, classification:c}
  classifications.insertOne(classification)
  if (!files[user].length){
    res.sendStatus(205)
  }else{
    res.sendStatus(200)
  }
  
});

MongoClient.connect(url, onConnect );