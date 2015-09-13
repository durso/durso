<?php
namespace library\dom\elements;
use library\dom\elements\element;
use library\dom\elements\components\text;
use library\dom\object;
use library\mediator\nodeElement;


/**
 * Description of paired
 *
 * @author durso
 */
abstract class paired extends element{


    public function addComponent(object $component){
        nodeElement::addChild($this, $component);
        if($this->isRendered){
            $this->updateJS('append',$component);
        }
    }
    public function append(element $component,$animation = "fadeIn"){
        nodeElement::addChild($this,$component);
        if($this->isRendered){
            $component->addClass("nodisplay");
            $this->updateJS($animation,$component);
        } 
    }
    
    public function removeComponent(object $component){
        if($this->isRendered){
            if($component instanceof text){
                $index = $this->indexOf($component);
                if($index !== false){
                    $component->removeValue();
                }
            } else {
                $this->updateJS('remove',$component->getUid());
            }
        }
        nodeElement::removeChild($this, $component);
    }
    
    public function removeChildren(){
        $children = $this->getChildren();
        foreach($children as $child){
            $this->removeComponent($child);
        }
    }
    
    public function remove(){
        $this->updateJS('remove');
        nodeElement::remove($this);
    }
    
    public function innerHTML(object $value){
        $this->removeChildren();
        $this->addComponent($value);
    }
    
    public function clear($component = false){
        if(!($component instanceof object)){
            if($component === false){
                $collection = $this->children();
            } else {
                $collection = $this->find($component);
            }
            if($collection instanceof elementCollection){
                $children = $collection->getCollection();
                foreach($children as $child){
                    $this->remove($child);
                }
            } else {
                $this->clear($collection);
            }
            $this->updateJS('clear');
        } else{
            nodeElement::removeChild($this, $component);
        }
    }
    /*
     * 
     * Render element to html
     * @return string
     */
    public function render(){
        $this->isRendered = true;
        $html = $this->openTag();
        if(nodeElement::hasChild($this)){
            foreach($this->getChildren() as $child){
                $html .= $child->render();
            }
        } 
        $html .= $this->closeTag();
        return $html;
    }
    
    protected function closeTag(){
        $close = "</".$this->tag.">";
        $this->html .= $close;
        return $close;
    }
    public function hasChild(){
        return nodeElement::hasChild($this);
    }
    public function getChildren($index = false){
        $children = nodeElement::getAllChildren($this);
        if($index === false)
            return $children;
        return $children[$index];
    }
    public function children($selector = false){
        return nodeElement::children($this,$selector);
    }
    public function find($selector){
        return nodeElement::find($this,$selector);
    }
    public function first($selector){
        return nodeElement::first($this,$selector);
    }
    
    public function changeText($value,$key){
        $child = $this->getChildren($key);
        $child->setValue($value);
    }
    public function appendText($value,$key){
        $child = $this->getChildren($key);
        $child->append($value);
    }

    public function addText($value){
        if(($index = $this->lastTextNode()) !== false){
            $this->appendText($value, $index);
        } else {
            $text = new text($value);
            $this->addComponent($text);
        }
    }
    
    protected function lastTextNode(){
        $children = $this->getChildren();
        if($children === false) return false;
        $index = count($children) - 1;
        if($children[$index] instanceof text) return $index;
        return false;
    }
    
    protected function indexOf(object $object){
        $children = $this->getChildren();
        if($children === false) return false;
        $index = 0;
        foreach($children as $child){
            if($child === $object)
                return $index;
            $index++;
        }
        return false;
    }
   
}
