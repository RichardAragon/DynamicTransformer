module HPPU(
    input [1:0] mode,  // 00: binary, 01: ternary, 10: full-precision
    input [15:0] a,
    input [15:0] b,
    output reg [31:0] result
);
    always @(*) begin
        case (mode)
            2'b00: begin
                // Binary operation
                result = (a[0] ? -1 : 1) * (b[0] ? -1 : 1);
            end
            2'b01: begin
                // Ternary operation
                result = (a == 16'h0000) ? 0 : ((a[15] ? -1 : 1) * (b == 16'h0000 ? 0 : (b[15] ? -1 : 1)));
            end
            2'b10: begin
                // Full-precision operation
                result = a * b;
            end
            default: result = 0;
        endcase
    end
endmodule
