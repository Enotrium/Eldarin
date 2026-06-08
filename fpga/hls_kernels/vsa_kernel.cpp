// VSA/HDC Kernel for FPGA
// ========================
// Hyperdimensional computing operations for Eldarin on FPGA.
// Implements binding, bundling, and similarity using bitwise logic.
//
// Reference: https://github.com/Enotrium/arthedain-1

#include <ap_int.h>
#include <hls_stream.h>

#define HD_DIM 4096
#define BINDING_XNOR  // Use XNOR for binary binding

typedef ap_uint<HD_DIM> hd_vector_t;
typedef ap_uint<16> popcount_t;

// Top-level VSA kernel
void vsa_hdc_top(
    hd_vector_t vec_a,
    hd_vector_t vec_b,
    hd_vector_t *bound_out,
    hd_vector_t *bundle_out,
    popcount_t *similarity_out,
    ap_uint<2> op  // 0=bind, 1=bundle, 2=similarity
) {
    #pragma HLS INTERFACE s_axilite port=vec_a
    #pragma HLS INTERFACE s_axilite port=vec_b
    #pragma HLS INTERFACE s_axilite port=bound_out
    #pragma HLS INTERFACE s_axilite port=bundle_out
    #pragma HLS INTERFACE s_axilite port=similarity_out
    #pragma HLS INTERFACE s_axilite port=op
    #pragma HLS INTERFACE s_axilite port=return

    switch(op) {
        case 0:
            // Binding via XNOR (for bipolar ±1, multiplication; for binary, XNOR)
            #ifdef BINDING_XNOR
            *bound_out = ~(vec_a ^ vec_b);  // XNOR
            #else
            *bound_out = vec_a ^ vec_b;     // XOR
            #endif
            break;

        case 1:
            // Bundling: element-wise majority (thresholded sum)
            // Simplified: OR for binary, sum > threshold for multi-vector
            *bundle_out = vec_a | vec_b;
            break;

        case 2:
            // Similarity: popcount of matching bits
            #ifdef BINDING_XNOR
            hd_vector_t match = ~(vec_a ^ vec_b);
            #else
            hd_vector_t match = vec_a ^ vec_b;
            #endif
            // Population count
            *similarity_out = 0;
            for (int i = 0; i < HD_DIM; i++) {
                #pragma HLS UNROLL factor=64
                *similarity_out += match[i];
            }
            break;
    }
}

// Streaming VSA for temporal sequences
void vsa_temporal_stream(
    hls::stream<hd_vector_t> &sequence_in,
    hls::stream<hd_vector_t> &bundled_out,
    ap_uint<16> seq_length
) {
    #pragma HLS INTERFACE axis port=sequence_in
    #pragma HLS INTERFACE axis port=bundled_out
    #pragma HLS INTERFACE s_axilite port=seq_length
    #pragma HLS INTERFACE s_axilite port=return

    hd_vector_t accumulator = 0;

    for (ap_uint<16> t = 0; t < seq_length; t++) {
        #pragma HLS PIPELINE II=1
        hd_vector_t vec = sequence_in.read();

        // Temporal permutation: cyclic shift by t
        hd_vector_t shifted;
        for (int i = 0; i < HD_DIM; i++) {
            shifted[i] = vec[(i + t) % HD_DIM];
        }

        accumulator = accumulator ^ shifted;  // XOR-based bundle
    }

    bundled_out.write(accumulator);
}

// Hamming distance kernel for tracking association
void vsa_hamming_distance(
    hd_vector_t query_hd,
    hd_vector_t *candidates_hd,
    popcount_t *distances,
    ap_uint<10> num_candidates
) {
    #pragma HLS INTERFACE m_axi port=candidates_hd offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=distances offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=query_hd
    #pragma HLS INTERFACE s_axilite port=num_candidates
    #pragma HLS INTERFACE s_axilite port=return

    for (ap_uint<10> c = 0; c < num_candidates; c++) {
        #pragma HLS PIPELINE II=1
        hd_vector_t candidate = candidates_hd[c];

        #ifdef BINDING_XNOR
        hd_vector_t match = ~(query_hd ^ candidate);
        #else
        hd_vector_t match = query_hd ^ candidate;
        #endif

        popcount_t count = 0;
        for (int i = 0; i < HD_DIM; i++) {
            #pragma HLS UNROLL factor=64
            count += match[i];
        }
        distances[c] = count;
    }
}