library ieee ;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity reg4 is
	port (
		D : in std_logic_vector(3 downto 0);
		C : in std_logic;
		E : in std_logic;
		R : in std_logic;
		Q : out std_logic_vector(3 downto 0)
	);
end reg4;

architecture behv of reg4 is
begin
	process (D, C, E, R) begin
		if R = '1' then
			Q <= "0000";
		elsif (C='1' and C'event) then
			if E = '1' then
				Q <= D;
			end if;
		end if;
	end process;
end behv;
