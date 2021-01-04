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
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Agent23 extends AbstractNegotiationParty {

    private List<Bid> bidOrder = null;
    private List<Issue> issues = null;
    private Map<Issue, Map<Value, Integer>> issueValueLabel = new HashMap<>();
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

    private double estimatedNP;
    private boolean didGetNP = false;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        if (hasPreferenceUncertainty()) {

            issues = userModel.getDomain().getIssues();
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

        if (t >= 0.3 && t < 0.9) {
//			didGetNP = true;
            Bid maxBid = maxUtilityBid;
            Bid minBid = minUtilityBid;
            double oppoMaxU = opponentModel.getUtility(lastOffer);
            double meMinU = getUtility(minBid);
            double meMaxU = getUtility(maxBid);
            double maxR = (oppoMaxU - meMinU)/(meMaxU - meMinU);
            estimatedNP = (1-maxR)/1.68 + maxR;
            log("estimatedNP: " + estimatedNP);
        }

        double lastOfferUtility = getUtility(lastOffer);

        if (t < 0.3) {
            double utilityMinusThresh = lastOfferUtility - threshold;
            if (utilityMinusThresh >= 0) {
                log("Time: " + t + "\tAccept Bid: " + lastOffer);
                return new Accept(getPartyId(), lastOffer);
            }
            if (utilityMinusThresh >= 0.1)
                return new Offer(getPartyId(), generateBidAroundUtility((lastOfferUtility + threshold) / 2, 0.03));
            if (utilityMinusThresh >= 0.25)
                return new Offer(getPartyId(), generateBidAroundUtility((lastOfferUtility + utilityMinusThresh * (10./9 * utilityMinusThresh + 7./18)), 0.02));
            return new Offer(getPartyId(), maxUtilityBid);
        }

        double opponentUtility = opponentModel.getUtility(lastOffer);
        if (lastOfferUtility >= threshold) {
            if (t < 0.95 && opponentUtility - lastOfferUtility > 0.2)
                return new Offer(getPartyId(), generateBidAgainstOpponent(opponentUtility, lastOfferUtility));
            log("Time: " + t + "\tAccept Bid: " + lastOffer);
            return new Accept(getPartyId(), lastOffer);
        }
        return new Offer(getPartyId(), generateBidAroundThreshold());

    }

    private void updateThreshold(double t) {
        if (t < 0.1) {
            threshold = 0.9;
        }else if (t < 0.3) {
            threshold =  0.9 - 0.5 * (t - 0.01);
        }else if (t < 0.4) {
            double p = 0.3 * (1 - estimatedNP) + estimatedNP;
            threshold =  0.9
                    - (0.9 - p) / (0.5 - 0.2) * (t - 0.2);
        }else if (t < 0.8) {
            double p1 = 0.3 * (1 - this.estimatedNP)
                    + this.estimatedNP;
            double p2 = 0.15 * (1 - this.estimatedNP)
                    + this.estimatedNP;
            this.threshold = p1 - (p1 - p2) / (0.9 - 0.5) * (t - 0.5);
        }else if (t < 0.95) {
            threshold = 0.86 - 0.35 * (t - 0.6);
        }else {
            threshold = 0.748 - 0.5 * (t - 0.92);
        }
    }

    private Bid generateBidAroundUtility(double utility, double tolerance) {
        Bid randomBid, selectedBid;
        double randomBidUtility;
        Map<Bid, Double> bidsUtilities = new HashMap<>();
        for (int i = 0; i < 2 * possibleBidsCount; i++) {
            randomBid = generateRandomBid();
            randomBidUtility = getUtility(randomBid);
            bidsUtilities.put(randomBid, Math.abs(randomBidUtility - utility));
        }
        List<Map.Entry<Bid, Double>> bidsUtilitiesList = new ArrayList<>(bidsUtilities.entrySet());
        bidsUtilitiesList.sort(Comparator.comparingDouble(Map.Entry::getValue));
        selectedBid = bidsUtilitiesList.get(0).getKey();
        if (bidsUtilitiesList.get(0).getValue() > tolerance)
            return maxUtilityBid;
        return  selectedBid;
    }

    private Bid generateBidAgainstOpponent(double opponentUtility, double agentUtility) {
        Bid randomBid;
        double distance = opponentUtility - agentUtility;
        double randomBidAgentUtility, randomBidOpponentUtility;
        for (int i = 0; i < 2 * possibleBidsCount; i++) {
            randomBid = generateRandomBid();
            randomBidAgentUtility = getUtility(randomBid);
            randomBidOpponentUtility = opponentModel.getUtility(randomBid);
            if (opponentUtility - randomBidAgentUtility < distance &&
                    opponentUtility - randomBidAgentUtility > 0 &&
                    randomBidOpponentUtility - agentUtility < distance &&
                    randomBidOpponentUtility - agentUtility >= 0)
                return randomBid;
        }
        return maxUtilityBid;
    }

    private Bid generateBidAroundThreshold() {
        double t = timeline.getTime();

        Bid randomBid, finalBid = null;
        double agentUtility, opponentUtility, metric;
        double maximumMetric = 0;

        if (t < 0.3) {
            Map<Bid, Double> bidsUtilities = new HashMap<>();
            int randomCount = possibleBidsCount > 200 ? 200: possibleBidsCount;
            for (int i = 0; i < randomCount; i++) {
                randomBid = generateRandomBid();
                agentUtility = getUtility(randomBid);
                bidsUtilities.put(randomBid, agentUtility);
            }
            List<Map.Entry<Bid, Double>> entries = new ArrayList<>(bidsUtilities.entrySet());
            entries.sort(Comparator.comparingDouble(Map.Entry::getValue));
            Random random = new Random();
            int randomIndex;
            if (t < 0.1)
                randomIndex = random.nextInt((int) (0.3 * randomCount)) + (int) (0.3 * randomCount);
            else
                randomIndex = random.nextInt((int) (0.5 * randomCount)) + (int) (0.3 * randomCount);
            return entries.get(randomIndex).getKey();

        }

        for (int i = 0; i < 2 * possibleBidsCount; i++) {
            randomBid = generateRandomBid();
            agentUtility = getUtility(randomBid);
            opponentUtility = opponentModel.getUtility(randomBid);
            metric = agentUtility * opponentUtility + t * opponentUtility + 0.85 * agentUtility;
            if (metric > maximumMetric) {
                finalBid = randomBid;
                maximumMetric = metric;
            }
        }
        if (finalBid == null)
            finalBid = maxUtilityBid;
        return finalBid;
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
                utility = realMaxUtility - 0.001;
            else if (utility < realMinUtility + 0.001)
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
