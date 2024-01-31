
import Chisel._

class AxiStream extends Bundle {
	val valid = Bool( INPUT)
	val ready = Bool(OUTPUT)
	val data  = UInt( INPUT, 8)
}

class AxiAddModule extends Module {
	val io = new Bundle {
		val axi_in  = new AxiStream()
		val axi_out = new AxiStream().flip
	}

	val reg_a = Reg(UInt())
	val reg_b = Reg(UInt())
	val state = Reg(init = UInt(0))

	io.axi_in.ready  := state === UInt(0) || state === UInt(1)
	io.axi_out.valid := state === UInt(2)
	io.axi_out.data  := reg_a + reg_b

	when (state === UInt(0) && io.axi_in.valid) {
		reg_a := io.axi_in.data
		state := UInt(1)
	}

	when (state === UInt(1) && io.axi_in.valid) {
		reg_b := io.axi_in.data
		state := UInt(2)
	}

	when (state === UInt(2) && io.axi_out.ready) {
		state := UInt(0)
	}
}

object design {
	def main(args: Array[String]): Unit = {
		if (args.length > 0)
			chiselMain(Array[String]("--targetDir", "target") ++ args, () => Module(new AxiAddModule()))
		else
			chiselMain(Array[String]("--targetDir", "target", "--backend", "v"), () => Module(new AxiAddModule()))
	}
}

