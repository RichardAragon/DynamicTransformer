module ControlUnit (
    input clk,
    input [31:0] performance_metric,
    output reg [1:0] mode
);
    always @(posedge clk) begin
        if (performance_metric < 32'd1000) begin
            mode <= 2'b10;  // Full-precision
        end else if (performance_metric < 32'd5000) begin
            mode <= 2'b01;  // Ternary
        end else begin
            mode <= 2'b00;  // Binary
        end
    end
endmodule
