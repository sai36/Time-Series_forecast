install.packages("smooth")
install.packages("forecast")
require(smooth)
require(forecast)

#We can provide the path for csv file
var <- read.csv("/path/to/csvfile")

result<-sma(var, order = 3, h = 10)
plot(result$actuals, col = 1)
lines(result$fitted, col = 2)
legend("topleft", col = 1:2,lty = 1, c("Actual", "Fitted"))
