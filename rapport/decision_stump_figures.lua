for i = 1, 3 do
	local n_estimators = i * 16
	tex.print(string.format([[
		\begin{figure}
			\centering
			\input{figures/3a/decision_stumps/decision_regions_%d.pgf}
			\caption{Régions de décision générées par AdaBoost avec des souches de décision et \numprint{%d} classifieurs}
			\label{fig:adaboost_decision_regions_with_decision_stumps_%d}
		\end{figure}
	]], n_estimators, n_estimators, n_estimators))
end