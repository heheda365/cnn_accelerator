#include "top.h"

int main(int argc, char const *argv[])
{
  ap_uint<64> in;
  ap_uint<64> out;
  bool doInit;
	unsigned int targetLayer;
  unsigned int targetMem;
	unsigned int targetInd;
  unsigned int targetThresh;
  ap_uint<64> val;
  unsigned int numReps = 2;

  BlackBoxJam(&in, &out, doInit,
		targetLayer, targetMem,
		targetInd, targetThresh, val, numReps);

    // BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		// unsigned int targetLayer, unsigned int targetMem,
		// unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps);
  return 0;
}
