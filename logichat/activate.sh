# This file must be used with "source activate.sh" *from bash*
# You cannot run it directly

_ok=true
for _f in activate.sh config.py en_basic.py formal.py parser.py tokens.py trainer_openwebtext.py; do
	test -f $_f || { echo "activate.sh must be included from within the directory containing it!" 2>&1; _ok=false; break; }
done
if $_ok; then
	if ! test -d .venv; then
		python3 -m venv .venv
	fi

	source .venv/bin/activate
	eval "$(declare -f deactivate | sed 's/^deactivate/_logichat_venv_deactivate/')"
	LOGICHAT_HOME="${VIRTUAL_ENV%/.venv}"

	pip_install_deps() {
		local pyver=`python3 --version | sed 's/.* //; s/\.[^.]*$//'`
		local site_path="${VIRTUAL_ENV}/lib/python${pyver}/site-packages"
		for arg; do
			test -d $site_path/${arg//-/_}-*.dist-info || (set -x; pip install $arg; )
		done
	}

	pip_install_deps z3-solver pcre2 numpy torch ptpython
	pip_install_deps transformers datasets tiktoken wandb tqdm
	pip_install_deps lxml mwparserfromhell pyphen

	deactivate() {
		unset LOGICHAT_HOME
		unset -f pip_install_deps
		_logichat_venv_deactivate
		unset -f _logichat_venv_deactivate
		unset -f deactivate
	}

	if [ -z "${VIRTUAL_ENV_DISABLE_PROMPT:-}" ] ; then
		export PS1="(LogiChat) ${PS1#${VIRTUAL_ENV_PROMPT}}"
		export VIRTUAL_ENV_PROMPT='(LogiChat) '
	fi
fi

unset _f _ok
