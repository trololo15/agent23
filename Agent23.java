package group23;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.BidRanking;
import genius.core.uncertainty.ExperimentalUserModel;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.CustomUtilitySpace;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

import java.util.*;
import java.util.stream.Collectors;

public class Agent23 extends AbstractNegotiationParty {

    private static boolean LOG_ENABLED = true;

    private List<Bid> bidOrder = null;
    private List<Issue> issues = null;
    private Map<Issue, Map<Value, Integer>> issueValueLabel;
    private Integer regFeatureDim = 0;
    private Integer possibleBidsCount = 1;
    private Matrix coefficient;
    private OpponentModel opponentModel;
    private double threshold;
    private Bid lastOffer;

    private Bid maxUtilityBid;
    private Bid minUtilityBid;
    private double realMaxUtility;
    private double realMinUtility;
    private double predMaxUtility;
    private double predMinUtility;

    private Bid lastReceivedBid;
    private double maxOpponentToAgentUtility;
    private boolean initialTimePass = false;
    private boolean maxOpponentToAgentBidAvailable = false;
    private double startTime;
    private double estimatedNashPoint = 0.8;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        if (hasPreferenceUncertainty()) {
            issues = userModel.getDomain().getIssues();
            issueValueLabel = new HashMap<>();

            for (Issue issue: issues) {
                IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
                List<ValueDiscrete> issueValues = issueDiscrete.getValues();
                Map<Value, Integer> valueLabel = new HashMap<>();
                for (int i = 0; i < issueValues.size(); ++i)
                    valueLabel.put(issueValues.get(i), i);
                regFeatureDim += issueValues.size();
                possibleBidsCount *= issueValues.size();
                issueValueLabel.put(issue, valueLabel);
            }

            BidRanking bidRanking = userModel.getBidRanking();
            bidOrder = bidRanking.getBidOrder();

            maxUtilityBid = bidOrder.get(bidOrder.size() - 1);
            minUtilityBid = bidOrder.get(0);

            // Prepare for training X
            double[][] oneHotEncodedBids = oneHotEncode(bidOrder);

            // Prepare for training y
            realMaxUtility = bidRanking.getHighUtility();
            realMinUtility = bidRanking.getLowUtility();

            double[] regressionValue = new double[bidOrder.size()];
            regressionValue[0] = realMinUtility;
            regressionValue[regressionValue.length - 1] = realMaxUtility;
            for (int i = 1; i < regressionValue.length - 1; ++i)
                regressionValue[i] = regressionValue[i - 1] + (realMaxUtility - realMinUtility) / (bidOrder.size() - 1);


            // Ridge Regression
            Matrix X = DenseMatrix.Factory.importFromArray(oneHotEncodedBids);
            Matrix y = DenseMatrix.Factory.importFromArray(regressionValue);
            Matrix I = DenseMatrix.Factory.eye(regFeatureDim, regFeatureDim);
            double lambda = 0.2;
            coefficient = X.transpose().mtimes(X).plus(I.times(lambda)).inv().mtimes(X.transpose()).mtimes(y.transpose());

            List<Bid> bids = new ArrayList<>();
            bids.add(minUtilityBid);
            bids.add(maxUtilityBid);
            Matrix Z = Matrix.Factory.importFromArray(oneHotEncode(bids));
            double[][] minMaxUtilities = Z.mtimes(coefficient).toDoubleArray();
            predMinUtility = minMaxUtilities[0][0];
            predMaxUtility = minMaxUtilities[1][0];

            // User Model test
            if (userModel instanceof ExperimentalUserModel) {
                log("You have given the agent access to the real utility space for debugging purposes.");
                ExperimentalUserModel e = (ExperimentalUserModel) userModel;
                AbstractUtilitySpace realUSpace = e.getRealUtilitySpace();
                log("User Model Test:\nEstimated Utility\tReal Utility");
                for (int i = 0; i < 10; i++) {
                    Bid randomBid = getUtilitySpace().getDomain().getRandomBid(new Random());
                    log(getUtility(randomBid) + "\t" + realUSpace.getUtility(randomBid));
                }
            }

            opponentModel = new OpponentModel();
        }
    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        double t = timeline.getTime();
        updateThreshold(t);

        if (lastOffer == null)
            return new Offer(getPartyId(), maxUtilityBid);

        double lastOfferUtility = getUtility(lastOffer);

        if (!maxOpponentToAgentBidAvailable) getMaxOpponentToAgent(t, 20.0 / 200.0, lastOfferUtility);
        lastReceivedBid = lastOffer;

        // 60轮之前不考虑opponent model
        if (t < 0.3) {
            log("Round " + timeline.getCurrentTime() + "\tTime " + t + "\tEstimatedUtility " + lastOfferUtility);
            double utilityMinusThresh = lastOfferUtility - threshold;
            // 如果对方给我们的utility大于等于threshold，则直接Accept
            if (utilityMinusThresh >= 0) {
                log("Time: " + t + "\tAccept Bid: " + lastOffer);
                return new Accept(getPartyId(), lastOffer);
            }
            // 如果对方给我们的utility非常接近threshold，则选择中间值附近的bid，如果选不出就Accept
            if (utilityMinusThresh >= -0.2) {
                double desiredUtility = (lastOfferUtility + 1) / 2;
                Bid selectedBid = generateBidAroundUtility(desiredUtility, -utilityMinusThresh * 0.3, lastOfferUtility, 1);
                if (selectedBid == null) {
                    log("Time: " + t + "\tAccept Bid: " + lastOffer);
                    return new Accept(getPartyId(), lastOffer);
                }
                return new Offer(getPartyId(), selectedBid);
            }

            // 如果对方给我们的utility较低时，生成较高utility的bid
            double desiredUtility = threshold - t * 0.3;
            Bid selectedBid = generateBidAroundUtility(desiredUtility, 0.02, 0.85, 1);
            if (selectedBid == null)
                return new Offer(getPartyId(), maxUtilityBid);
            return new Offer(getPartyId(), selectedBid);
        }

        // 60轮之后尽量给对手高的utility，同时慢慢降低自己的utility
        double opponentUtility = opponentModel.getUtility(lastOffer);
        log("Round " + timeline.getCurrentTime() + "\tTime " + t + "\tEstimatedUtility " + lastOfferUtility + "\tOpponent Utility " + opponentUtility);

        if (lastOfferUtility >= threshold && opponentUtility >= threshold - 0.2) {
            log("Time: " + t + "\tAccept Bid: " + lastOffer);
            return new Accept(getPartyId(), lastOffer);
        }
        return new Offer(getPartyId(), generateBidMaximumOpponentUtility(threshold, 0.02));
    }

    private void getMaxOpponentToAgent(double time, double timeLast, double bidUtility) {
        if (bidUtility > maxOpponentToAgentUtility)
            maxOpponentToAgentUtility = bidUtility;

        if (initialTimePass) {
            if (time - startTime > timeLast) {
                double maxOpponentToAgentBidRatio = (maxOpponentToAgentUtility - realMinUtility) / (realMaxUtility - realMinUtility);
                estimatedNashPoint = (1 - maxOpponentToAgentBidRatio) / 1.7 + maxOpponentToAgentBidRatio; // 1.414 是圆，2是直线
                maxOpponentToAgentBidAvailable = true;
                log("Estimated Nash Point: " + estimatedNashPoint);
            }
        } else {
            if (lastReceivedBid != lastOffer) {
                initialTimePass = true;
                startTime = time;
            }
        }
    }

    private void updateThreshold(double t) {
        if (t < 0.1)
            threshold = 0.95;
        else if (t < 0.3)
            threshold = 0.95 - (t - 0.1) * 0.25;
        else if (t < 0.5) {
            double p1 = 0.3 * (1 - estimatedNashPoint) + estimatedNashPoint;
            threshold = 0.9 - (0.9 - p1) / (0.5 - 0.2) * (t - 0.2);
        }
        else if (t < 0.9) {
            double p1 = 0.3 * (1 - estimatedNashPoint) + estimatedNashPoint;
            double p2 = 0.15 * (1 - estimatedNashPoint) + estimatedNashPoint;
            threshold = p1 - (p1 - p2) / (0.9 - 0.5) * (t - 0.5);
        }
        else if (t < 0.95) {
            double p2 = 0.15 * (1 - estimatedNashPoint) + estimatedNashPoint;
            double p3 = 0.05 * (1 - estimatedNashPoint) + estimatedNashPoint;
            threshold = p2 - (p2 - p3) / (0.95 - 0.9) * (t - 0.9);
        }
        else if (t < 0.99) {
            double p3 = 0.05 * (1 - estimatedNashPoint) + estimatedNashPoint;
            double p4 = -0.35 * (1 - estimatedNashPoint) + estimatedNashPoint;
            threshold = p3 - (p3 - p4) / (0.99 - 0.95) * (t - 0.95);
        }
        else
            threshold = -0.4 * (1 - estimatedNashPoint) + estimatedNashPoint;
    }

    /**
     * 生成目标值desiredUtility附近的bid
     * @param desiredUtility 想要得到的bid的utility值
     * @param optimalRange bid的最佳utility范围(desiredUtility-optimalRange, desiredUtility+optimalRange)
     * @param minUtility bid的utility最低可能值
     * @param maxUtility bid的utility最高可能值
     * @return 如果找到bid则返回bid，否则返回null
     */
    private Bid generateBidAroundUtility(double desiredUtility, double optimalRange, double minUtility, double maxUtility) {
        Bid randomBid, backupBid1 = null, backupBid2 = null;
        double randomBidUtility;

        for (int i = 0; i < 2 * possibleBidsCount; ++i) {
            randomBid = generateRandomBid();
            randomBidUtility = getUtility(randomBid);
            // 如果找到utility附近的bid，则直接返回
            if (Math.abs(randomBidUtility - desiredUtility) < optimalRange)
                return randomBid;
            // 如果没有找到，优先选择大于utility且小于maxUtility的最小bid
            if (randomBidUtility > desiredUtility && randomBidUtility < maxUtility) {
                backupBid1 = randomBid;
                maxUtility = randomBidUtility;
            }
            // 如果还是没有找到，则选择小于utility且大于minUtility的最大bid
            else if (backupBid1 == null && randomBidUtility < desiredUtility && randomBidUtility > minUtility) {
                backupBid2 = randomBid;
                minUtility = randomBidUtility;
            }
        }
        if (backupBid1 != null)
            return backupBid1;
        return backupBid2;
    }

    /**
     * 生成目标值desiredUtility附近，且对手utility最高的bid
     * @param desiredUtility 想要得到的bid的utility值
     * @param optimalRange bid的最佳utility范围(desiredUtility-optimalRange, desiredUtility+optimalRange)
     * @return 如果找到bid则返回bid，否则扩大optimalRange范围直到找到为止
     */
    private Bid generateBidMaximumOpponentUtility(double desiredUtility, double optimalRange) {
        Bid randomBid;
        double agentUtility, opponentUtility, distance;
        Map<Bid, Double> bidUtilityMap = new HashMap<>();
        for (int i = 0; i < 2 * possibleBidsCount; ++i) {
            randomBid = generateRandomBid();
            agentUtility = getUtility(randomBid);
            distance = Math.abs(agentUtility - desiredUtility);
            if (distance <= optimalRange) {
                opponentUtility = opponentModel.getUtility(randomBid);
                bidUtilityMap.put(randomBid, opponentUtility);
            }
        }
        if (bidUtilityMap.size() == 0)
            return generateBidMaximumOpponentUtility(desiredUtility, optimalRange + 0.01);

        List<Map.Entry<Bid, Double>> bidUtilityList = new ArrayList<>(bidUtilityMap.entrySet());
        bidUtilityList.sort(Comparator.comparingDouble(Map.Entry::getValue));
        RandomCollection<Bid> bidRandomCollection = new RandomCollection<>();
        for (int i = 0; i < bidUtilityList.size() * 0.1; ++i) {
            Map.Entry<Bid, Double> entry = bidUtilityList.get(bidUtilityList.size() - i - 1);
            bidRandomCollection.add(entry.getValue(), entry.getKey());
        }
        return bidRandomCollection.next();
    }

    @Override
    public void receiveMessage(AgentID sender, Action act) {
        if (act instanceof Offer) {
            lastOffer = ((Offer) act).getBid();
            opponentModel.update(lastOffer);
        }
    }

    @Override
    public AbstractUtilitySpace estimateUtilitySpace() {
        return new RegressionUtilitySpace(getDomain());
    }

    @Override
    public String getDescription() {
        return "Group 23 Negotiation Agent";
    }

    private void log(Object msg) {
        if (LOG_ENABLED)
            System.out.println(msg);
    }

    // convert bids to labels
    private int[][] bidsLabelEncode(List<Bid> bids) {
        int[][] labelEncoded = new int[bids.size()][issues.size()];
        for (int i = 0; i < bids.size(); ++i) {
            for (int j = 0; j < issues.size(); ++j) {
                Value value = bids.get(i).getValue(issues.get(j));
                labelEncoded[i][j] = issueValueLabel.get(issues.get(j)).get(value);
            }
        }
        return labelEncoded;
    }

    private double[][] oneHotEncode(List<Bid> bids) {
        int[][] labelsArray = bidsLabelEncode(bids);
        List<Integer> issueSizes = issues.stream().map(issue -> ((IssueDiscrete) issue).getValues().size()).
                collect(Collectors.toList());

        List<Integer> sizeAccumulation = new ArrayList<>();
        sizeAccumulation.add(0);
        for (Integer issueSize: issueSizes)
            sizeAccumulation.add(issueSize + sizeAccumulation.get(sizeAccumulation.size() - 1));
        sizeAccumulation.remove(0);

        double[][] oneHotArray = new double[labelsArray.length][sizeAccumulation.get(sizeAccumulation.size() - 1)];

        for (int i = 0; i < labelsArray.length; ++i)
            for (int j = 0; j < labelsArray[i].length; ++j)
                oneHotArray[i][sizeAccumulation.get(j) - labelsArray[i][j] - 1] = 1;

        return oneHotArray;
    }

    public class RandomCollection<E> {
        private final NavigableMap<Double, E> map = new TreeMap<>();
        private final Random random;
        private double total = 0;

        public RandomCollection() {
            this(new Random());
        }

        RandomCollection(Random random) {
            this.random = random;
        }

        public RandomCollection<E> add(double weight, E result) {
            if (weight <= 0) return this;
            total += weight;
            map.put(total, result);
            return this;
        }

        public E next() {
            double value = random.nextDouble() * total;
            return map.higherEntry(value).getValue();
        }
    }

    private class RegressionUtilitySpace extends CustomUtilitySpace {

        RegressionUtilitySpace(Domain dom) {
            super(dom);
        }

        @Override
        public double getUtility(Bid bid) {
            double k = (realMaxUtility - realMinUtility) / (predMaxUtility - predMinUtility);
            double b = realMinUtility - k * predMinUtility;

            List<Bid> bids = new ArrayList<>();
            bids.add(bid);
            Matrix X = Matrix.Factory.importFromArray(oneHotEncode(bids));
            double utility = X.mtimes(coefficient).toDoubleArray()[0][0];
            utility = k * utility + b;
            if (utility > realMaxUtility)
                utility = realMaxUtility - new Random().nextDouble() * 0.1;
            else if (utility < realMinUtility + new Random().nextDouble() * 0.1)
                utility = realMinUtility;
            return utility;
        }
    }

    private class OpponentModel {
        Map<Issue, Double> issueWeights;
        Map<Issue, Map<Value, Double>> issueOptionValues;
        private Map<Issue, Map<Value, Integer>> issueValueFreq;
        private boolean weightsUpdated = false;

        OpponentModel() {
            issueWeights = new HashMap<>();
            issueOptionValues = new HashMap<>();
            issueValueFreq = new HashMap<>();

            for (Issue issue: issues) {
                IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
                List<ValueDiscrete> issueValues = issueDiscrete.getValues();
                Map<Value, Integer> valueFreq = new HashMap<>();
                for (Value issueValue: issueValues)
                    valueFreq.put(issueValue, 0);
                issueValueFreq.put(issue, valueFreq);
            }
        }

        void update(Bid lastOffer) {
            for (Map.Entry<Issue, Map<Value, Integer>> entry: issueValueFreq.entrySet()) {
                Value lastOfferValue = lastOffer.getValue(entry.getKey());
                entry.getValue().computeIfPresent(lastOfferValue, (k, v) -> v + 1);
            }
            weightsUpdated = false;
        }

        // lazy update model
        private boolean updateModel() {
            int bidsCount = 0;
            for (Map.Entry<Issue, Map<Value, Integer>> entry: issueValueFreq.entrySet()) {
                Issue issue = entry.getKey();
                Map<Value, Integer> valueFreq = entry.getValue();
                if (bidsCount == 0) {
                    bidsCount = valueFreq.values().stream().mapToInt(Integer::intValue).sum();
                    if (bidsCount == 0)
                        return false;
                }

                double issueWeight = 0;
                for (int freq: valueFreq.values())
                    issueWeight += Math.pow(freq / (double) bidsCount, 2);
                issueWeights.put(issue, issueWeight);

                Map<Value, Double> optionValue = new HashMap<>();
                List<Map.Entry<Value, Integer>> valueFreqList = new ArrayList<>(valueFreq.entrySet());
                valueFreqList.sort((o1, o2) -> o2.getValue() - o1.getValue());

                int n = valueFreqList.size();
                for (int i = 0; i < n; ++i)
                    optionValue.put(valueFreqList.get(i).getKey(), (n - i) / (double)n);
                issueOptionValues.put(issue, optionValue);
            }
            final double totalWeights = issueWeights.values().stream().mapToDouble(Double::doubleValue).sum();
            issueWeights.replaceAll((k, v) -> v / totalWeights);

            weightsUpdated = true;
            return true;
        }

        double getUtility(Bid bid){
            if (!weightsUpdated)
                if (!updateModel())
                    return 0;

            double utility = 0;
            double optionValue, issueWeight;
            for (Issue issue: bid.getIssues()) {
                optionValue = issueOptionValues.get(issue).get(bid.getValue(issue));
                issueWeight = issueWeights.get(issue);
                utility += issueWeight * optionValue;
            }
            return utility;
        }
    }
}
