
library ieee ;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity regcell is
port(	input:	in std_logic_vector(2 downto 0);
	enable:	in std_logic;
	Q:	out std_logic_vector(2 downto 0)
);
end regcell;

-- no architecture here - xst will recognize it as blackbox module --

