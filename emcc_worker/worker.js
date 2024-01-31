
importScripts('example.js');

onmessage = (function(e){
	request = e.data[0];
	Module.example_cfun(request);
	postMessage([request]);
});

