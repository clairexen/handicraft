library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
	port (
		clk : in std_logic;
		rst : in std_logic;
		up3 : in std_logic;
		dn2 : in std_logic;
		cnt : out std_logic_vector(7 downto 0)
	);
end entity;

architecture rtl of top is
	signal state : std_logic_vector(7 downto 0);
begin
	process (clk) begin
		if (rising_edge(clk)) then
			if (up3) then
				state <= std_logic_vector(unsigned(state) + 3);
			end if;
			if (dn2) then
				state <= std_logic_vector(unsigned(state) - 3);
			end if;
			if (rst) then
				state <= x"00";
			end if;
		end if;
	end process;

	cnt <= state;
end architecture;

