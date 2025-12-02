# IBS Boxplot Generator
# Usage: Rscript boxplot.R <input_csv_path> <output_pdf_path>
args = commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop("Usage: Rscript boxplot.R <input_csv_path> <output_pdf_path>")
}

input_path = args[1]
output_path = args[2]

# Check if ggplot2 is installed, if not, install it
if (!require("ggplot2", quietly = TRUE)) {
    cat("ggplot2 package not found. Installing ggplot2...\n")
    install.packages("ggplot2", repos = "https://cran.rstudio.com/")
    library("ggplot2")
}

# Read the CSV file
df_IBS = read.csv(input_path)

# Process the data
df_IBS$N = as.factor(df_IBS$N)

# Convert G_type labels to proper names matching manuscript figures
df_IBS$G_type[which(df_IBS$G_type=="linear")] = "Linear"
df_IBS$G_type[which(df_IBS$G_type=="quad")] = "Quadratic"
df_IBS$G_type[which(df_IBS$G_type=="sin")] = "Sinusoidal"
df_IBS$G_type = factor(df_IBS$G_type, levels = c("Linear", "Quadratic", "Sinusoidal"))

# Ensure Method labels are correct
# Support both old (Nonlinear) and new (NonLinear1, NonLinear2) naming schemes
df_IBS$Method[which(df_IBS$Method=="Nonlinear")] = "NonLinear1"
df_IBS$Method = factor(df_IBS$Method, levels = c("CoxPH", "NonLinear1", "NonLinear2"))

# Define colors matching g(z) and Lambda plots:
# NonLinear1 = red (matching red dotted line in plots)
# NonLinear2 = green (matching green dashed line in plots)
# CoxPH = blue (distinct from red and green)
method_colors = c("CoxPH" = "blue", "NonLinear1" = "red", "NonLinear2" = "green")

# Create a paired boxplot matching manuscript style
p = ggplot(df_IBS, aes(x = N, y = IBS, fill = Method)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.8, outlier.size = 1) +
  scale_fill_manual(values = method_colors, name = "Method") +
  labs(title = "", x = "n", y = "IBS") +
  facet_wrap(~G_type, ncol = 3) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    legend.position = "top",
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10),
    panel.spacing = unit(1, "lines")
  )

# Save the plot
ggsave(output_path, plot = p, width = 12, height = 5, units = "in")
cat("IBS boxplot saved to:", output_path, "\n")
