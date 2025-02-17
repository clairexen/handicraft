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

#ifdef ENABLE_WDROID_ENGINE_3
template struct WordleDroidEngine<3>;
#endif

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

bool WordleDroidGlobalState::parseNextCommand()
{
	using namespace std::literals;

	currentCommand = {};
	hideSubCommandPrefix = {};
	parsedCurrentCommand.clear();
	promptRewriteEnabled = false;
	assert(nextEngine == nullptr);

	// --------------------------
	// Fetch next command line

	if (commandStack.empty()) {
		char buffer[1024], *p = buffer;
		engine->prPrompt();
		if (fgets(buffer, 1024, stdin) == nullptr) {
			engine->pr('\r');
			commandStack.push_back("-exit");
			return true;
		}
		if (char *cursor = strchr(buffer, '\n'); cursor != nullptr)
			*cursor = 0;
		currentCommand = p;
		if (currentCommand.starts_with(".."))
			engine->prReplaceLastLine();
		else
			promptRewriteEnabled = true;
	} else {
		currentCommand = commandStack.back();
		commandStack.pop_back();
		if (!currentCommand.starts_with("..")) {
			engine->prPrompt();
			engine->pr(currentCommand);
			engine->prNl();
			promptRewriteEnabled = true;
		}
	}

	// --------------------------
	// Split command line into commands

	std::string_view arg, cmd = currentCommand;

	while (cmd.starts_with('.')) {
		hideSubCommandPrefix = ".."sv;
		cmd = cmd.substr(1);
	}

	bool longCommand = cmd.starts_with("--") || cmd.starts_with("++");
	cmd = cmd.substr(longCommand ? 1 : 0);

	if (!longCommand)
	{
		std::vector<std::string_view> cmds;
		cmds.push_back(cmd);

		while (1) {
			auto k = cmds.back().find_first_of(" \t\n\r\f\v");
			if (k == std::string_view::npos)
				break;
			auto first = cmds.back().substr(0, k);
			auto second = cmds.back().substr(k+1);
			cmds.pop_back();
			if (!first.empty())
				cmds.push_back(first);
			if (!second.empty())
				cmds.push_back(second);
		}

		if (cmds.size() > 1) {
			for (auto it = cmds.rbegin(); it != cmds.rend(); it++) {
				std::string c(*it);
				while (!c.starts_with(hideSubCommandPrefix))
					c = "."s + c;
				commandStack.push_back(c);
			}
			return true;
		}
	}

	// --------------------------
	// Split commands into command name, optional arg,
	// and optional pargs with names and optional args

	parsedCurrentCommand.emplace_back(cmd, ""sv);

	while (parsedCurrentCommand.back().second.empty())
	{
		cmd = parsedCurrentCommand.back().first;

		int argBegin = 0;
		while (argBegin < cmd.size()) {
			char ch = cmd[argBegin++];
			if (argBegin == 1 && ch == '+') continue;
			if ('a' <= ch && ch <= 'z') continue;
			if ('A' <= ch && ch <= 'Z') continue;
			if ('0' <= ch && ch <= '9') continue;
			if (ch == '_' || ch == '-') continue;
			argBegin--;
			break;
		}

		if (argBegin == cmd.size())
			break;

		parsedCurrentCommand.back().first = cmd.substr(0, argBegin);
		parsedCurrentCommand.back().second = cmd.substr(argBegin);

		if (longCommand || parsedCurrentCommand.back().second.empty())
			break;

		auto plusBegin = parsedCurrentCommand.back().second.find('+');
		if (plusBegin == std::string_view::npos)
			break;

		cmd = parsedCurrentCommand.back().second.substr(plusBegin);
		arg = parsedCurrentCommand.back().second.substr(0, plusBegin);
		parsedCurrentCommand.back().second = arg;
		parsedCurrentCommand.emplace_back(cmd, ""sv);
	}

	return false;
}

void WordleDroidGlobalState::executeNextCommand()
{
	using namespace std::literals;

	if (parseNextCommand())
		return;

	auto [cmd, arg] = parsedCurrentCommand.front();
	// auto pargs = parsedCurrentCommand | std::views::drop(1);

	// --------------------------
	// Simple global commands

#ifdef ENABLE_WDROID_ENGINE_3
	if (cmd == "-3"sv) {
		delete engine;
		engine = new WordleDroidEngine3(this, arg);
		return;
	}
#endif

#ifdef ENABLE_WDROID_ENGINE_4
	if (cmd == "-4"sv) {
		delete engine;
		engine = new WordleDroidEngine4(this, arg);
		return;
	}
#endif

#ifdef ENABLE_WDROID_ENGINE_5
	if (cmd == "-5"sv) {
		delete engine;
		engine = new WordleDroidEngine5(this, arg);
		return;
	}
#endif

#ifdef ENABLE_WDROID_ENGINE_6
	if (cmd == "-6"sv) {
		delete engine;
		engine = new WordleDroidEngine6(this, arg);
		return;
	}
#endif

	if (cmd == "-exit"sv) {
		if (showKeys)
			engine->prShowKeyboard();
		delete engine;
		engine = nullptr;
		return;
	}

	if (cmd == "-K"sv) {
		showKeys = engine->boolArg(arg);
		return;
	}

	if (cmd == "-M"sv) {
		showMasks = engine->boolArg(arg);
		return;
	}

	if (cmd == "-R"sv) {
		refineMasks = engine->boolArg(arg);
		return;
	}

	if (cmd == "-system"sv && !arg.empty()) {
		std::string systemCommandBuf(arg.substr(1));
		system(systemCommandBuf.c_str());
		return;
	}

	if (engine->vGetWordLen() == 0) {
		if (promptRewriteEnabled)
			engine->prReplaceLastLine();
		commandStack.push_back(currentCommand);
		commandStack.push_back("-5");
		return;
	}

	// --------------------------
	// Defer to engine->vExecuteCommand()
	// and engine->vExecuteBasicCommand()

	int newEngineCount = 0;
	bool gotNewEngine = false;

	auto handleNextEngine = [&](bool rc) -> bool {
		if (nextEngine == nullptr) {
			gotNewEngine = false;
			return rc;
		}
		delete engine;
		engine = nextEngine;
		nextEngine = nullptr;
		gotNewEngine = true;
		newEngineCount++;
		return rc;
	};

got_new_engine:
	if (newEngineCount >= 10) {
		engine->pr(std::format("Error executing command '{}': "
				"Stuck in engine reload loop.\n", currentCommand));
		return;
	}

	if (handleNextEngine(engine->vExecuteNextCommand()))
		return;
	if (gotNewEngine)
		goto got_new_engine;
	if (handleNextEngine(engine->vExecuteBasicCommand()))
		return;
	if (gotNewEngine)
		goto got_new_engine;

	engine->pr(std::format("Error executing command '{}'! Try -h for help.\n", currentCommand));
}

int WordleDroidGlobalState::main(int argc, const char **argv)
{
	for (int i = argc-1; i > 0; i--)
		commandStack.push_back(argv[i]);
	while (engine)
		executeNextCommand();
	return 0;
}

REG_WDROID_CMDS(WordleDroidEngine, "-reset")

int main(int argc, const char **argv)
{
	WordleDroidGlobalState state;
	return state.main(argc, argv);
}
