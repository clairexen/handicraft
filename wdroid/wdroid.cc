// The New York Times "WordleBot" is behnd a paywall.  :/
// So I wrote my own "WordleDroid" which I can run locally.
//
// Copyright (C) 2025  Claire Xenia Wolf <claire@clairexen.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#include "wdroid.hh"

#ifdef ENABLE_WDROID_ENGINE_4
template struct WordleDroidEngine<4>;
#endif

#ifdef ENABLE_WDROID_ENGINE_5
template struct WordleDroidEngine<5>;
#endif

#ifdef ENABLE_WDROID_ENGINE_6
template struct WordleDroidEngine<6>;
#endif

void AbstractWordleDroidEngine::pr(char c) const {
	if (globalState && globalState->outfile.is_open())
		globalState->outfile << c;
	else
		std::cout << c;
}

void AbstractWordleDroidEngine::pr(const std::string &s) const {
	if (globalState && globalState->outfile.is_open())
		globalState->outfile << s;
	else
		std::cout << s;
}

void AbstractWordleDroidEngine::prFlush() const {
	if (globalState && globalState->outfile.is_open())
		globalState->outfile << std::flush;
	else
		std::cout << std::flush;
}

bool WordleDroidGlobalState::executeCommand(const char *p, const char *arg, bool noprompt)
{
	using namespace std::string_literals;

	if (p == nullptr) {
		char buffer[1024], *p = buffer;
		engine->prPrompt();
		if (fgets(buffer, 1024, stdin) == nullptr) {
			engine->pr('\r');
			return executeCommand("-exit", nullptr, false);
		}
		if (char *cursor = strchr(buffer, '\n'); cursor != nullptr)
			*cursor = 0;
		if (p[0] == '-' && p[1] == '-') {
			engine->prReplaceLastLine();
			return executeCommand(p+1, nullptr, true);
		}
		return executeCommand(buffer, nullptr, true);
	}

	if (p[0] == '-' && p[1] == '-')
		return executeCommand(p+1, arg, true);

	if (arg == nullptr) {
		if (const char *s = strchr(p, '='); s != nullptr) {
			char *buffer = strdup(p);
			char *cursor = buffer + (s - p);
			*(cursor++) = 0;
			bool ret = executeCommand(buffer, cursor, noprompt);
			free(buffer);
			return ret;
		}
	}

	if (!noprompt) {
		engine->prPrompt();
		engine->pr(p);
		if (arg) {
			engine->pr('=');
			engine->pr(arg);
		}
		engine->prNl();
	}

#ifdef ENABLE_WDROID_ENGINE_4
	if (p == "-4"s) {
		delete engine;
		engine = new WordleDroidEngine4(this, arg);
		return true;
	}
#endif

#ifdef ENABLE_WDROID_ENGINE_5
	if (p == "-5"s) {
		delete engine;
		engine = new WordleDroidEngine5(this, arg);
		return true;
	}
#endif

#ifdef ENABLE_WDROID_ENGINE_6
	if (p == "-6"s) {
		delete engine;
		engine = new WordleDroidEngine6(this, arg);
		return true;
	}
#endif

	if (p == "-exit"s) {
		if (showKeys)
			engine->prShowKeyboard();
		delete engine;
		engine = nullptr;
		return true;
	}

	if (p == "-K"s) {
		showKeys = engine->boolArg(arg);
		return true;
	}

	if (p == "-M"s) {
		showMasks = engine->boolArg(arg);
		return true;
	}

	if (p == "-R"s) {
		refineMasks = engine->boolArg(arg);
		return true;
	}

	if (engine->vGetWordLen() == 0) {
		engine->prReplaceLastLine();
		executeCommand("-5", nullptr);
		return executeCommand(p, arg);
	}

	AbstractWordleDroidEngine *nextEngine = nullptr;
	if (!engine->vExecuteCommand(p, arg, nextEngine)) {
		delete nextEngine;
		if (arg == nullptr)
			printf("Error executing command '%s'! Try -h for help.\n", p);
		else
			printf("Error executing command '%s' with arg '%s'! Try -h for help.\n", p, arg);
		return false;
	}

	if (nextEngine) {
		delete engine;
		engine = nextEngine;
	}

	return true;
}

int WordleDroidGlobalState::main(int argc, const char **argv)
{
	for (int i=1; engine != nullptr && i<argc; i++) {
		if (!executeCommand(argv[i], nullptr))
			return 1;
	}

	while (engine != nullptr)
		executeCommand(nullptr, nullptr);

	return 0;
}

int main(int argc, const char **argv)
{
	WordleDroidGlobalState state;
	return state.main(argc, argv);
}
