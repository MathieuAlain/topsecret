for n_estimators = 16, 50, 16 do
	tex.print(([[
		\begin{figure}
			\centering
			\input{figures/3a/decision_stumps/decision_regions_%d.pgf}
			\caption{Régions de décision générées par AdaBoost avec des souches de décision et \numprint{%d} classifieurs}
			\label{fig:adaboost_decision_regions_with_decision_stumps_%d}
		\end{figure}
	]]):format(n_estimators, n_estimators, n_estimators))
end
