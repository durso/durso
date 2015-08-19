<?php
/**
 * This class is the common superclass of all DOM elements.
 *
 * 
 * @author durso
 */
namespace library\dom;
use library\tree\node;
use library\tree\nodeFactory;
use library\dom\javascript;
use library\mediator\nodeElement;

abstract class object {

    /*
     * 
     * @var string html to be rendered 
     */
    protected $html = "";
    /*
     * 
     * @var boolean if element is rendered
     */
    protected $isRendered = false;
    /*
     * 
     * @var node list
     */
    protected $node;
    
    public function __construct(){
        $this->node = nodeFactory::create($this); 
    }
    public function getNode(){
        return $this->node;
    }
    public function setNode(node $node){
        $this->node = $node;
    }
    public function getParent(){
        if($this->node->hasParent()){
            return $this->node->getParent()->getValue();
        }
    }
    public function siblings($selector = false){
        return nodeElement::siblings($this,$selector);
    }
    public function parents($selector = false){
        return nodeElement::parents($this,$selector);
    }
    public function siblingsIndex(){
        return nodeElement::siblingsIndex($this);
    }

    /*
     * 
     * Changes value of isRendered
     * @param boolean $boolean
     * @return void
     */
    public function setRenderFlag($boolean){
        $this->isRendered = $boolean;
    }
    /*
     * 
     * Render element to html
     * @return string
     */
    public function render(){
        $this->isRendered = true;
        return $this->html;
    }
    public function hasTag(){
        return false;
    }
    protected function updateJS($method,$value = false,$key = false){
        if(javascript::isActive()){
            javascript::update($this,$method,$value,$key);
        }
    }

    public function __toString() {
        return $this->render();
    }

    
}
