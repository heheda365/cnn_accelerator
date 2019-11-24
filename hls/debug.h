#pragma once

#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#include <assert.h>

string hexFromInt(int value, unsigned precision) {
	unsigned hex_digits = precision/4;
	if (precision%4 > 0)
		hex_digits += 1;

	if (value < 0)
		value = (1 << precision) + value;

	string result = "";
	for (unsigned d = 0; d < hex_digits; d++) {
		unsigned temp = value & 0xF;
		value = value >> 4;
		stringstream ss;
		ss << hex << temp;
		result = ss.str() + result;
	}
	
	return result;
}

template <	unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void monitor(
	stream<ap_uint<Cin*Ibit> > & in,
	const char* filename,
	unsigned reps = 1)
{
	ofstream fileout(filename);

	for (unsigned rep = 0; rep < reps; rep++) {
#ifdef MISC_DEBUG
		cout << "-----------------------------------" << endl;
#endif
		for (unsigned h = 0; h < Din; h++) {
			for (unsigned w = 0; w < Din; w++) {

				ap_uint<Cin*Ibit> temp = in.read();
				in.write(temp);

				string line = "";
				for (unsigned c = 0; c < Cin; c++) {
					line = hexFromInt( temp( (c+1)*Ibit - 1, c*Ibit ), Ibit ) + "_" + line;
#ifdef MISC_DEBUG
					cout << temp( (c+1)*Ibit - 1, c*Ibit ) << " ";
#endif
				}
				fileout << "0x" << line;
			}
            fileout << endl;
#ifdef MISC_DEBUG
			cout << endl;
#endif
		}
	}
    fileout.close();
}