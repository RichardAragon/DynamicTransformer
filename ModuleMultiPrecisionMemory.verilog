module MultiPrecisionMemory (
    input clk,
    input [1:0] mode,  // 00: binary, 01: ternary, 10: full-precision
    input [7:0] address,
    input [31:0] data_in,
    output reg [31:0] data_out,
    input we
);
    reg [1:0] memory [0:255];  // Binary and ternary storage
    reg [31:0] full_precision_memory [0:255];  // Full-precision storage

    always @(posedge clk) begin
        if (we) begin
            case (mode)
                2'b00: memory[address] <= data_in[1:0];
                2'b01: memory[address] <= data_in[1:0];  // Similar for ternary
                2'b10: full_precision_memory[address] <= data_in;
            endcase
        end else begin
            case (mode)
                2'b00: data_out <= {30'b0, memory[address]};
                2'b01: data_out <= {30'b0, memory[address]};
                2'b10: data_out <= full_precision_memory[address];
            endcase
        end
    end
endmodule
