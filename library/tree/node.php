<?php
/**
 * Node
 *
 * @author durso
 */
namespace library\tree;
use library\dom\dom;



abstract class node {
   
    /**
     *
     * @var node parent node  
     */
    protected $parent = null;
    /**
     *
     * @var mixed value of the node  
     */
    protected $value;
    
    public function __construct($value = null){
        if(!is_null($value)){
            $this->value = $value;
        }
    }
    
    public function setValue($value){
	$this->value = $value;
    }
    /*
     * Get the value of the node
     * @return mixed
     */
    public function getValue(){
	return $this->value;
    }
    /*
     * 
     * Check if the node has parent
     * @return boolean
     */
    public function hasParent(){
        return !is_null($this->parent);
    }
     /*
     * 
     * Set the node parent
     * @param node $parent
     * @return void
     */
    public function setParent(node $node = null){
	$this->parent = $node;
    }
     /*
     * 
     * Get the parent node
     * @return node
     */
    public function getParent(){
       return $this->parent;
    }
     /*
     * 
     * Get ancestor by index
     * Example:
     * Given a tree with 3 levels, to get the root node from the bottom-most node $x:
     * <code>
     * <?php
     *      $rootNode = $x->getAncestor(3);
     * ?>
     * </code> 
     * @param int $index
     * @return mixed
     */
    public function getAncestor($index){
        $node = $this;
        $i = 1;
        while (true) {
            $node = $node->getParent();
            if(!$node){
                break;
            }
            if($i == $index){
                return $node;
            }
            $i++;
        }
        return false;
    }
    
    /*
     * 
     * Get all ancestors from the node
     * @return array
     */
    public function getAncestors(){
        $ancestors = array();
        $node = $this;
        while (true) {
            $parent = $node->getParent();
            if(!$parent){
                break;
            }
            $ancestors[] = $parent;
            $node = $parent;
        }
        return $ancestors;

    }

    /*
     * 
     * Get node siblings
     * @return mixed
     */
    public function getSiblings($self = false){
       $parent = $this->getParent();
       if($parent){
            $children = $parent->getChildren();
            if($self) return $children;
            return utils::array_remove($children,$this);  
       }
       return false;
    }
    
   
    /*
     * 
     * Check if node has siblings
     * @return boolean
     */
    public function hasSiblings(){
       return count($this->getSiblings()) > 0;
    }

    public function __wakeup(){
        $this->value->setNode($this);
    }
}
