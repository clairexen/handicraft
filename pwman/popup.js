
var seed = "defaultseed";

function update()
{
	if (document.getElementById("pw").value == "") {
		document.getElementById("pass").value = "";
		return;
	}

	if (document.getElementById("pw").value == "*SET*SEED*") {
		seed = window.prompt("Entry (or copy&paste) seed string");
		localStorage.setItem("pwman-seed", seed);
		window.location.reload();
		return;
	}

	var hash = CryptoJS.SHA256(seed + ":" + document.getElementById("pw").value + ":" + document.getElementById("st").value);
	var hh = String(CryptoJS.SHA256(seed + document.getElementById("pw").value)).slice(0, 2);
	var chars = "abcdefghkmnpqrstuvwxyz" +
	            "ABCDEFGHKMNPQRSTUVWXYZ" +
	            "23456789$%^&*()!/+=?";
	var pass = "";

	for (var i = 0; i < 16; i++) {
		var charIndex = parseInt("0x" + String(hash).slice(2*i, 2*i+2)) % 64;
		pass += chars.slice(charIndex, charIndex+1);
	}

	document.getElementById("hh").innerText = hh;
	document.getElementById("pass").value = pass;
	document.getElementById("pass").select();
}

function onload()
{
	function setSite(site)
	{
		if (site == "") {
			site = "localhost";
		} else {
			site = site.split(".");
			if (site.length >= 2)
				site = site[site.length-2] + "." + site[site.length-1];
			else
				site = site[site.length-1];
		}

		document.getElementById("st").value = site;
		document.getElementById("pw").focus();
	}

	if (chrome && chrome.tabs && chrome.tabs.getCurrent)
		chrome.windows.getCurrent({ populate: true }, function(win) {
			var url = "";
			var parser = document.createElement('a');
			for (var i in win.tabs) {
				if (win.tabs[i].highlighted)
					parser.href = win.tabs[i].url;
			}
			setSite(parser.hostname);
		});
	else
		setSite(window.location.hostname);

	try {
		var stored_seed = localStorage.getItem("pwman-seed");
		if (stored_seed) {
			document.getElementById("hh").innerText = "*";
			seed = stored_seed;
		}
	} catch (err) {
		/* ignore error */
	}

	document.getElementById("pw").addEventListener("change", update);
	document.getElementById("st").addEventListener("change", update);
}

window.addEventListener("load", onload);

