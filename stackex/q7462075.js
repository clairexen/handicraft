
function getRemainingStackSize()
{
	var i = 0;
	function stackSizeExplorer() {
		i++;
		stackSizeExplorer();
	}

	try {
		stackSizeExplorer();
	} catch (e) {
		return i;
	}
}

var baselineRemStackSize = getRemainingStackSize();
var largestSeenStackSize = 0;

function getStackSize()
{
	var sz = baselineRemStackSize - getRemainingStackSize();
	if (largestSeenStackSize < sz)
		largestSeenStackSize = sz;
	return sz;
}

function ackermann(m, n)
{
	if (m == 0) {
		console.log("Stack Size: " + getStackSize());
		return n + 1;
	}
	
	if (n == 0)
		return ackermann(m - 1, 1);

	return ackermann(m - 1, ackermann(m, n-1));
}

function main()
{
	var m, n;

	for (var m = 0; m < 4; m++)
	for (var n = 0; n < 5; n++)
		console.log("A(" + m + ", " + n + ") = " + ackermann(m, n));
	console.log("Deepest recursion: " + largestSeenStackSize + " (" +
			(baselineRemStackSize-largestSeenStackSize) + " left)");
}

main();

